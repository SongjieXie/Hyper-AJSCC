# Deep Learning-Based Adaptive Joint Source-Channel Coding using Hypernetwork

This repository is the PyTorch implementation of Hyper-AJSCC, as proposed in the paper [My Paper Title](https://arxiv.org/abs/2030.12345).

## Requirements

The codes are compatible with the packages:

1. pytorch 2.0.1

2. torchvision 0.15.2

3. numpy 1.25.2

4. tensorboardX 2.6.2.2

The code can be run on the datasets such as [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) . One should download the datasets in a directory (e.g., `./data/`) and change the root parameter in `datasets/dataloader.py`, e.g., 

```python
root = r'./data/
```

Additionally, since there is no official implementation of Generalized divisive normalization layer (GDN) in PyTorch, one may need to use the PyTorch implementation at [GitHub - jorge-pessoa/pytorch-gdn: PyTorch implementation of the Generalized divisive normalization non-linearity layer](https://github.com/jorge-pessoa/pytorch-gdn#generalized-divisive-normalization-layer) and put the corresponding source code in `models/pytorch_gdn.py`.

## Training

To train the model(s) for image transmission, run this command:

```shell
python main.py --train 1 --sc 16  --type recon --epoches 800 --lr 1e-4 --step 400
```

To train the model(s) for image classification, run this command:

```shell
python main.py --train 1 --sc 8 --type task --epoches 400 --lr 1e-3 --step 150
```

Then, the log files `logs_recon\` and `logs_task\` records the information such as the values of loss in the training stage. And the trained models are automatically stored at the directory `trained_models\`. 

## Evaluation

To evaluate the model trained for image transmission , run:

```shell
python main.py --train 0 --sc 16  --type recon --epoches 800 --lr 1e-4 --step 400
```

Or similarly run the following command to evaluate the models trained for image classification,

```shell
python main.py --train 0 --sc 8 --type task --epoches 400 --lr 1e-3 --step 150
```



Then, the evaluation results are recorded in the tensorboard files in directories `result_recon\` or `result_task\`

## Citation

## Updates