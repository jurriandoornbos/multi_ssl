Using Lightly SSL for multispectal pretraining
with some additional linear heads for segmentation and prediction.

Self-Supervised Learning on images often requires higher batch-sizes, and is built around heavy augmentation and the original image with some from of reconstruction task. This learns very good transferrable, generalizable weights. (minimal labels needed for task-transfer)

Although we are missing pretrained baselines and architectures for Multispectral imagery.
The other channels that are being used should create some more interesting and better insights, but also makes it incompatible with existing DL vision models.

So here are some baseline models, scripts, adjustments and tools to finetune a minor layer on minimal data.

LightlySSL is the main SSL software framework we will use:

## Installation
https://docs.lightly.ai/self-supervised-learning/getting_started/install.html

```bash
mamba/conda create --name lightly python==3.9 
conda activate lightssl
```

Do a manual pytorch w/cuda install first (Lightly is compatible with PyTorch and PyTorch Lightning v2.0+!)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then we are ready for lightly and we want tifffile for our more complex image files and cv2 for the data augments
```bash
pip install lightly tifffile opencv-python-headless wandb
```

## Training
Everything has been implemented trhough PytorchLightning, an example script is presented in
`multissl/train/train.py`:
With parameters to select training method and model architecture.

First implemented options are the FastSiam training method (default: `--ssl_method fastsiam`) and backbones ResNet18, ResNet50 (default: `--backbone resnet18`).

```bash
python multissl/train/train.py --input_dir D:\\Jurrian\chipped_512 --backbone resnet50 --epochs 2
```
python multissl/train.py --input_dir D:\Jurrian\chipped_336 --backbone resnet50 --num_workers 15

python multissl/train.py --input_dir ../msdata/data/chipped_336 --num_workers 0


python multissl/train.py --input_dir ../msdata/data/chipped_512 --num_workers 0 --backbone swin-tiny --epochs 2

# Run a comparison of various SSL methods?
nahhh
1. FastSiam

# Variation in backbone:
1. Resnet18 (11M) 
2. Resnet50 (22M)
3. Swin TF (33M)


# Findings for vineseg task:

R50+4.5M head findings?

1. More complex backbone does not yield better performance

R18+3.5M head findings:

2. Pretraining does not increase any-shot test accuracy compared to End-to-End training

3. Pretraining increases stability in training, the frozen backend has already found a non-overfitting optimum

4. Pretraining does not increase generalization of accuracy

5. Dataset variation is the most important to increase
    HOW to showcase this best?


  







