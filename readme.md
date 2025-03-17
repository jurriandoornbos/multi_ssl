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

## Variation in backbone:
1. Resnet18 (11M) 
2. Resnet50 (22M)
3. Swin TF (27.5M)


Writing tasks todo:
intro - context modern deep learning, lack of multispectral

related works - sota ssl, sota ssl remote sensing-sats, then what is left?

methods - Training dataset, SSL FastSiam, training params, segheads, evaluation tasks
results - results!

discussion - complexity of model impacts, resource use of training, limitations in evaluation

conclusion/next steps

programming tasks todo: (for vineseg)

1. train FastSIAM:
    - resnet18 (done)
    - resnet50
    - swin-tiny (done)
2. make resnet18-seg, resnet50-seg and swin-tiny-seg heads 
    - 2a: make swin-tiny-seg head
    - 2b: make resnet50-seg (prolly resnset18 works)
3. train 1-shot example:
    - 3a make 1-shot dataset
    - 3b train:
        - resnet18-seg pretrained
        - resnet18-seg untrained
        - resnet50-seg pretrained
        - resnet50-seg untrained
        - swintiny-seg pretrained
        - swintiny-seg untrained

4. show validation loss per model
    - pretraining acts as a great regularizer!

5. Therefore: 
    - accuracy results pretraining last-model VS low-validation-loss selected model on all datasets
    
    - visualize results pretraining 1-shot performance of last-model VS low-validation-loss selected model (2imgs per model, 6 models = 12 imgs)

How about a randomforest classifier? with 1 label

0-shot performance:


- semi supervised tuning along........
