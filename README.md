# [HAWAII] HAar Wavelet transform-based contrastive learning framework for the Image-to-Image translation 🏖
[paper]([http://taesung.me/ContrastiveUnpairedTranslation/](https://ieeexplore.ieee.org/document/10945777))  

This is the official Pytorch Implementation of HAWAII by Soobin Park^\dagger$, Seohyun Yoo^\dagger$, Nabin Jeong and Enju Cha*, accepted to IEEE Access.(^\dagger$ contributed equally, * corresponding author)

![Image](https://github.com/user-attachments/assets/8509d4de-1101-4974-9731-7c250d07dfb3)

![Image](https://github.com/user-attachments/assets/79873780-e207-48e2-961d-304238addf89){: width="200" height="200"}


We proposed an innovative approach to applying contrastive learning to unpaired Image-to-Image translation using the Haar wavelet trasnform, called HAWAII. 
HAWAII leverages the Haar wavelet transform to define typical and non-typical features across frequency bands, which correspond to mutual information and domain-specific features in previous contrastive learning methods.
By using these features, the generator learns to distinguish which information should be preserved or modified during translation. 
Moreover, contrastive learning is applied not only to the generator but also to the discriminator, enhancing its ability to distinguish between source and target domain features. This regularization strategy allows both components to more effectively identify domain-relevant information. 
Extensive experiments demonstrate that the proposed method achieves state-of-the-art performance across multiple benchmarks.



## Usage

## Requirements

## Datasets
Download the Horse$\rightarrow$Zebra dataset via below command. Please refer to [CycleGAN]() for more details.
```bash
bash ./datasets/download_cut_dataset.sh horse2zebra
```

## Training

## Test



### Acknowledge
Our implementation builds on [CUT](). We are also grateful to the contributors of [PyTorch-FID, KID](), and [Dino-Struct Dist.]().
```


### Acknowledgments
We thank Allan Jabri and Phillip Isola for helpful discussion and feedback. Our code is developed based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation,  [drn](https://github.com/fyu/drn) for mIoU computation, and [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch/) for the PyTorch implementation of StyleGAN2 used in our single-image translation setting.
