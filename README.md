# [HAWAII] HAar Wavelet transform-based contrastive learning framework for the Image-to-Image translation 🏖
[paper]([http://taesung.me/ContrastiveUnpairedTranslation/](https://ieeexplore.ieee.org/document/10945777))  

This is the official Pytorch Implementation of HAWAII by Soobin Park^\dagger$, Seohyun Yoo^\dagger$, Nabin Jeong and Enju Cha*, accepted to IEEE Access.(^\dagger$ contributed equally, * corresponding author)

![Image](https://github.com/user-attachments/assets/8509d4de-1101-4974-9731-7c250d07dfb3)

<img src="https://github.com/user-attachments/assets/79873780-e207-48e2-961d-304238addf89" width="200" height="200"/>
<img src="https://github.com/user-attachments/assets/06ae5c6b-ff72-41d4-8802-fad2febd7425" width="200" height="200"/>
<img src="https://github.com/user-attachments/assets/0a51b297-4cfc-48cb-98f8-7a88d6bc74c0" width="200" height="200"/>
<img src="https://github.com/user-attachments/assets/513259cb-5e5a-4194-a6de-3e5aea02bc58" width="200" height="200"/>


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
Our implementation builds on [CUT](). We are also grateful to the contributors of [pytorch-fid](https://github.com/mseitzer/pytorch-fid), [KID](), and [Dino-Struct Dist.]().
