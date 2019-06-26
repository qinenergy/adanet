# ADA-Net
Tensorflow implementation

[Semi-Supervised Learning by Augmented Distribution Alignment](https://arxiv.org/abs/1905.08171)  Qin Wang, Wen Li, Luc Van Gool (2019 Under reivew)




### Requirements
```
pip3 install tensorflow-gpu==1.13.1
pip3 install tensorpack==0.9.1
pip3 install scipy
```
###  Train and Eval ADA-Net on ConvLarge
#### Prepare dataset
```
cd convlarge
python3 cifar10.py --data_dir=./dataset/cifar10/ --dataset_seed=1
```

#### Train and Eval ADA-Net on Cifar10 ConvLarge

```
CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py --dataset=cifar10 --data_dir=./dataset/cifar10/ --log_dir=./log/cifar10aug/ --num_epochs=2000 --epoch_decay_start=1500 --aug_flip=True --aug_trans=True --dataset_seed=1
CUDA_VISIBLE_DEVICES=0 python3 test_cifar.py --dataset=cifar10 --data_dir=./dataset/cifar10/ --log_dir=<path_to_log_dir> --dataset_seed=1
```

Here are the error rates we get using the above scripts:

| Seed 1 | Seed 2 | Seed 3 |
| -------- | -------- | -------- |
| 8.61%     | 8.89%     | 8.65%     |

### Train and Eval ADA-Net on ImageNet ResNet
Download our imagenet labeled/unlabeled split from [this link](https://drive.google.com/open?id=1ZeG4Qr1z65Fwj9m8uffUWG1aymX14HZ3), put them in ./resnet

```
cd resnet
python3 ./adanet-resnet.py --data <path_to_your_imagenet_files> -d 18  --mode resnet --batch 256 --gpu 0,1,2,3
```


### Acknowledgement
+ ConvLarge code is based on Takeru Miyato's [tf implementation](https://github.com/takerum/vat_tf). 
+ ResNet code is based on [Tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/ResNet)'s supervised imagenet training scripts.

### License
MIT

### Citing this work
```
@article{wang2019semi,
  title={Semi-Supervised Learning by Augmented Distribution Alignment},
  author={Wang, Qin and Li, Wen and Van Gool, Luc},
  journal={arXiv preprint arXiv:1905.08171},
  year={2019}
}
```
