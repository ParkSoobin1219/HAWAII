import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .haar2d import Haar
import torch.nn.functional as F
import torchvision.transforms as transforms

class HAWAIIModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_NCE_D', type=float, default=0.5)
        parser.add_argument('--lambda_idt', type=float, default=1.0)
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--freq_separation', type=util.str2bool, nargs='?', const=True, default=False, help="Set 'True' if u wanna use freq sepa model")
        parser.add_argument('--batch_size_dec', type=int, default=10, help='input batch size')
        parser.add_argument('--netF_D', type=str, default='mlp')
        parser.add_argument('--D_feat_layers', type=str, default='1', help='compute SimSiam or NCE loss on which layers of the Discriminator')
        parser.add_argument('--use_sa_layers', type=util.str2bool, default=False)
        parser.add_argument('--sa_blocks', type=str, default=[])
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D', 'G_GAN', 'D_real', 'D_fake', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        if opt.freq_separation:
            self.loss_names += ['total_nce_loss_D']
            # ysh0912
            self.visual_names += ['real_A_ll']
            self.visual_names += ['real_A_lh']
            self.visual_names += ['real_A_hl']
            self.visual_names += ['real_A_hh']
            if opt.lambda_NCE_D > 0.0:
                self.D_feat_layers = [int(i) for i in self.opt.D_feat_layers.split(',')]

        if opt.lambda_idt>0 and self.isTrain:
            self.loss_names += ['crit_idt']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        if opt.freq_separation:
            if opt.lambda_NCE_D > 0.0:
                self.netF_D = networks.define_F(opt.input_nc, 'mlp', opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        self.n_NCE_layers_enc = len(self.nce_layers)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data, data_dec):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        bs_per_gpu_dec = data_dec["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data, data_dec)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.real_A_dec = self.real_A_dec[:bs_per_gpu_dec]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
                if self.opt.lambda_NCE_D>0.0:
                    self.optimizer_F_D = torch.optim.Adam(self.netF_D.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                    self.optimizers.append(self.optimizer_F_D)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
            if self.opt.lambda_NCE_D>0.0:
                self.optimizer_F_D.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
            if self.opt.lambda_NCE_D>0.0:
                self.optimizer_F_D.step()
    def set_input(self, input, input_dec):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if input_dec is not None:
            self.real_A_dec = input_dec['A' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        self.real_A_ll, self.real_A_lh, self.real_A_hl, self.real_A_hh = Haar(self.real_A, self.device)
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        z1 = self.real_B #[1, 3, 256, 256] make it 64(1 batch)batch_size
        hor_flip = transforms.RandomHorizontalFlip()
        ver_flip = transforms.RandomVerticalFlip()
        zebra = hor_flip(ver_flip(z1))
        ###########################
        # Load a RealB and Real As
        ###########################
        RealB = self.real_B
        RealB_aug = hor_flip(ver_flip(RealB))
        RealAs  = self.real_A_dec

        ###################################
        # Discriminator Feature Extraction
        ###################################
        feats_RealB = self.netD(RealB, layers=self.D_feat_layers) # [1,C,H,W]
        feats_RealB_aug = self.netD(RealB_aug, layers=self.D_feat_layers) # [1,C,H,W]
        feats_RealAs = self.netD(RealAs, layers=self.D_feat_layers) # [N,C,H,W]

        ##################
        # HAAR TRANSFORM
        ##################
        feats_RealB_high = []
        feats_RealB_low = []
        for idx, feat in enumerate(feats_RealB):
            ll,lh,hl,hh = Haar(feat, self.device)
            hs = torch.cat((lh, hl, hh), 1) # channel wise
            feats_RealB_high.append(hs) # ([N,C,H,W]) ; (1,64,128,128])
            feats_RealB_low.append(ll)

        feats_RealB_aug_high = []
        feats_RealB_aug_low = []
        for idx, feat in enumerate(feats_RealB_aug):
            ll,lh,hl,hh = Haar(feat, self.device)
            hs = torch.cat((lh,hl,hh),1) # channel wise
            feats_RealB_aug_high.append(hs)
            feats_RealB_aug_low.append(ll)

        feats_RealAs_high = []
        feats_RealAs_low = []
        for idx, feat in enumerate(feats_RealAs):
            ll,lh,hl,hh = Haar(feat, self.device)
            hs = torch.cat((lh, hl, hh), 1) # channel wise
            feats_RealAs_high.append(hs) # ([N,C,H,W]) ; ([10,64,128,128])
            feats_RealAs_low.append(ll)
        
        ########################3
        # Pass the MLP Networks
        ########################
        feat_RealB_pool = self.netF_D(feats_RealB_high)
        feat_RealB_aug_pool = self.netF_D(feats_RealB_aug_high)
        feat_RealAs_pool = self.netF_D(feats_RealAs_high)

        ##########################
        # Calculate Cross-Entropy
        ##########################
        total_nce_loss_D = 0.0
        layer_idx = 0
        n_layers = len(self.D_feat_layers)
        for f_q, f_p, f_n, nce_layer in zip(feat_RealB_pool, feat_RealB_aug_pool, feat_RealAs_pool, self.D_feat_layers):
            loss = self.CrossEntropy(f_q, f_p, f_n) * self.opt.lambda_NCE
            total_nce_loss_D += loss
            layer_idx += 1

        self.loss_total_nce_loss_D = total_nce_loss_D/n_layers
        
        ########################
        # Add D_nce to D_gan
        ########################
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + self.opt.lambda_NCE_D * self.loss_total_nce_loss_D
        return self.loss_D

    def CrossEntropy(self, f_q, f_p, f_n):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        # calculate logits
        l_pos = f_q*f_p
        l_pos = l_pos.permute(1,2,0) # C,nc, 1
        l_pos = l_pos.flatten(0,1) # C*nc, 1
        l_neg = f_q*f_n
        l_neg = l_neg.permute(1,2,0) # C, nc, N
        l_neg = l_neg.flatten(0,1) # C*nc, N
        # calculate crossentropy
        out = torch.cat((l_pos, l_neg),dim=1)/self.opt.nce_T
        loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=f_q.device))
        return loss.mean()
    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_crit_idt = self.criterionIdt(self.idt_B, self.real_B) * self.opt.lambda_NCE * self.opt.lambda_idt
            loss_NCE_both = (self.loss_NCE + self.loss_crit_idt) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        ### Haar
        feat_k_l = []
        for idx, feat in enumerate(feat_k):
            feat_k_ll, _, _, _ = Haar(feat, self.device)
            feat_k_l.append(feat_k_ll)
        feat_q_l = []
        for idx, feat in enumerate(feat_q):
            feat_q_ll, _, _, _ = Haar(feat, self.device)
            feat_q_l.append(feat_q_ll)
        ### Haar
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
