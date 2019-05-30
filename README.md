# ADA-Net
Tensorflow implementation for our paper 
[Semi-Supervised Learning by Augmented Distribution Alignment](https://arxiv.org/abs/1905.08171)  Qin Wang, Wen Li, Luc Van Gool




#### Requirements
tensorflow-gpu (Tested on 1.10.0 and 1.13.1)
scipy 0.19.0 (for ZCA whitening required by the original ConvLarge Implementation in VAT)
tensorpack (for ImageNet experiments)

####  Train and Eval ADA-Net on ConvLarge
##### Prepare dataset
```
cd convlarge
python3 svhn.py --data_dir=./dataset/svhn/
```

##### Training with ADA-Net on SVHN

```
CUDA_VISIBLE_DEVICES=0 python3 train_svhn.py --dataset=svhn --data_dir=./dataset/svhn/ --log_dir=./log/svhnaug/ --num_epochs=120 --epoch_decay_start=80 --aug_trans=True
```

##### Evaluation of the trained model

```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset=svhn --data_dir=./dataset/svhn/ --log_dir=<path_to_log_dir>
```

#### Train and Eval ADA-Net on ImageNet
Download our imagenet labeled/unlabeled split from [this link](https://drive.google.com/open?id=1ZeG4Qr1z65Fwj9m8uffUWG1aymX14HZ3), put them in ./resnet

```
python3 ./adanet-resnet.py --data <path_to_your_imagenet_files> -d 18  --mode resnet --batch 256 --gpu 0,1,2,3
```

### Acknowledgement
ConvLarge code is based on Takeru Miyato's tf implementation of [VAT](https://github.com/takerum/vat_tf). 
ResNet code is based on [Tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/ResNet)'s supervised imagenet training scripts.

## License
MIT