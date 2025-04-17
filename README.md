# [HAWAII] HAar Wavelet transform-based contrastive learning framework for the Image-to-Image translation 🏖

[paper](https://ieeexplore.ieee.org/document/10945777)

This is the official Pytorch implementation of HAWAII by Soobin Park $^\dagger$, Seohyun Yoo $^\dagger$, Nabin Jeong and Enju Cha*, accepted to IEEE Access.
($^\dagger$ These authors contributed equally, * corresponding author)

![Image](https://github.com/user-attachments/assets/8509d4de-1101-4974-9731-7c250d07dfb3)

<img src="https://github.com/user-attachments/assets/79873780-e207-48e2-961d-304238addf89" width="200" height="200"/>
<img src="https://github.com/user-attachments/assets/06ae5c6b-ff72-41d4-8802-fad2febd7425" width="200" height="200"/>
<img src="https://github.com/user-attachments/assets/0a51b297-4cfc-48cb-98f8-7a88d6bc74c0" width="200" height="200"/>
<img src="https://github.com/user-attachments/assets/513259cb-5e5a-4194-a6de-3e5aea02bc58" width="200" height="200"/>


We proposed an innovative approach to apply contrastive learning to unpaired Image-to-Image translation using the Haar wavelet trasnform, called HAWAII. 
HAWAII leverages the Haar wavelet transform to define typical and non-typical features across frequency bands, which correspond to mutual information and domain-specific features in previous contrastive learning based methods.
By using typical features, the generator effectively learns to distinguish which information should be preserved or modified during translation. 
The non-typical features are utilized in contrastive learning applied not to the generator but to the discriminator, enhancing its ability to distinguish between source and target domain features. 
This regularization strategy helps both the generator and the discriminator more effectively identify which features should be preserved or translated by taking frequency components into account.
Extensive experiments demonstrate that the proposed method achieves state-of-the-art performance across multiple benchmarks.


## Requirements
Get a suitable conda environment named 'hawaii' using:
```bash
conda env create -f environment.yaml
```

## Datasets
Download the Horse2Zebra dataset via below command. Please refer to [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for more details.
```bash
bash ./datasets/download_cut_dataset.sh horse2zebra
```

## Usage
Train the HAWAII model:
```bash
sh scripts/run_train.sh
```
Test the HAWAII model:
```bash
sh scripts/run_test.sh
```

## Pretrained Models
We provide the pretrained models on Horse2Zebra [here]().

## Acknowledge
Our implementation builds on [CUT](https://github.com/taesungp/contrastive-unpaired-translation/tree/master). We are also grateful to the contributors of [pytorch-fid](https://github.com/mseitzer/pytorch-fid), [KID](https://github.com/alpc91/NICE-GAN-pytorch), and [Dino-Struct Dist.](https://github.com/omerbt/Splice).
