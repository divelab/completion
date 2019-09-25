# Deep Adversarial Learning for Multi-Modality Missing Data Completion

Deep Adversarial Learning for Multi-Modality Missing Data Completion can be used to generate missing modality using the existing modality. 

Detailed informationis provided in [paper] (https://dl.acm.org/citation.cfm?id=3219963).

![model](./utils/data.pdf)

## Citation

If using this code, please cite our paper.

```
@inproceedings{cai2018deep,
  title={Deep adversarial learning for multi-modality missing data completion},
  author={Cai, Lei and Wang, Zhengyang and Gao, Hongyang and Shen, Dinggang and Ji, Shuiwang},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1158--1166},
  year={2018},
  organization={ACM}
}
```

## System requirement

#### Programming language
Python 

#### Python Packages
tensorflow (CPU) or tensorflow-gpu (GPU), numpy, h5py, progressbar, PIL, scipy

## Prepare data

We use ANDI dataset in our paper. Please download the dataset and process the data following the paper. The processed data shoud be saved in h5 format. Both training and testing h5 file should contains three keys ('label', 'mri', 'pet'). The shape of 'mri' and 'pet' in h5 file should be NxDxHxWxC. In adni dataset, we process the data as D=H=W=64, C=1. The shape of 'label' in h5 file shoud be Nx1.

## Configure the network

All network hyperparameters are configured in main.py.

#### Training

max_step: how many iterations or steps to train

test_step: how many steps to perform a mini test or validation

save_step: how many steps to save the model

summary_step: how many steps to save the summary

trade_off: trade of MSE loss and adversarial loss

#### Data

data_dir: data directory

train_data: h5 file for training

valid_data: h5 file for validation

test_data: h5 file for testing

batch: batch size

channel: input image channel number

height, width: height and width of input image

#### Debug

logdir: where to store log

modeldir: where to store saved models

sampledir: where to store predicted samples, please add a / at the end for convinience

model_name: the name prefix of saved models

reload_step: where to return training

test_step: which step to test or predict

random_seed: random seed for tensorflow

#### Network architecture

network_depth: how deep of the U-Net including the bottom layer

start_channel_num: the number of channel for the first conv layer

## Training and Testing

#### Start training

After configure the network, we can start to train. Run
```
python main.py
```

#### Acknowledge

Part of code borrows from [PixelDCL](https://github.com/HongyangGao/PixelDCN). Thanks for their excellent work!
