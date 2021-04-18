# ComputerVision

## TP1
TBD

## TP2
TBD

## TF - Bacteria Segmentation
### Computer Vision
TBD

### Deep Learning
- This Part of the TF was developed using the Pytorch framework.
- Regarding the CNN model used in this work, we choose a custumized UNET architecture where we change its backbone to use a VGG model.

#### How to run:

First of all, you need to install the NVIDIA Automatic Mixed Precision (Amp).
You can find more information about it and also how to install it in this GitHub repo: https://github.com/NVIDIA/apex

To train the model for bacteria segmentation, just open the terminal inside the VGG folder and type:
```python
python3.8 train.py
```
