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



End of day:

Semi-Supervised with EMA student-teacher seems to fail:
    Heavy augs too heavy? - Nope was an error with MASKEING
    Hyperparameters?


Adapters version works well. Could do with balancing the adapter VS segmenter: overfit on one over the other at the moment.
Also implement with semi-heavy augmentations?
yeah, doesnt work amazingly sadly

Something with an EMA adapter? student-teacher vibe
Perhapssss - Works okay I think

# Implement FastSIAM with channel dropout augs?
- Retrain R18, and SWIN model

R50+4.5M head findings?

1. More complex backbone does not yield better performance

R18+3.5M head findings:

2. Pretraining does not increase any-shot test accuracy compared to End-to-End training

3. Pretraining increases stability in training, the frozen backend has already found a non-overfitting optimum

4. Pretraining does not increase generalization of accuracy

5. Dataset variation is the most important to increase
    HOW to showcase this best?
Make datasets with 5 samples in the training set: none in the eval set
    train5_q
    train5_e
    train5_v

train on e: eval on e score: 
            eval on all score:

train on eqv: eval on e score: the same/similar to only e
              eval on all score: better to only e

6. It is all about varying the examples PER class
    That's why we do minimal label-semi supervised learning




# Findings for vineseg task Part 2:

Semi-supervised on the backbone is nice.
Although it does not seem to train the correct things from the dataset yet.
I think it could be overfitting on the lower layers of the bbone? This method could work better with a SWIN?


Trading it for no-pretraining causes it to heavily overfit and maintain low confidence in its scores.


Fully supervised training on A:

test on valA, B and C results in poor performance

Semi supervised training on A (with raw img from B and C) is my new goal:
It sometimes works -> especially quite wel for B, but not valA and C interestingly

Ofcourse we are getting good scores. with F1 of around 70: that is basically predicting all-black
Adjusting for class weights doesnt seem to change much.

Teacher-student Semi slightly better
Adapter Semi not quite good enough

Interestingly: dataset size doesnt seem to matter -Batchnorm to groupnorm first tho
1 or 5 10 or all samples aint changing much in  results. Just much longer training.


New Heads that better work on the input. It seems that the original head kinda suced ass?


Writing tasks todo:
intro - context modern deep learning, lack of multispectral

related works - sota ssl, sota ssl remote sensing-sats, then what is left?

methods - Training dataset, SSL FastSiam, training params, segheads, evaluation tasks
results - results!

discussion - complexity of model impacts, resource use of training, limitations in evaluation

conclusion/next steps

programming tasks todo: (for vineseg)

1. train FastSIAM:
    - resnet18 (done-ish) - new augs todo (v3)
    - resnet50
    - swin-tiny (in progress)
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

0-shot performance:


- semi supervised tuning along........
