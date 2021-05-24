# ComputerVision

## TP1
TBD

## TP2
TBD

## TF - Bacteria Segmentation
### Computer Vision
We developed a pipeline in which the steps taken by the approach are mainly derived from traditional computer vision techniques:
- Changes in color space;
- Histogram equalization;
- Thresholds;
- Morphological operations;
- Feature extraction (using HOG);
- Supervised Learning (with Support Vector Machine, i.e. SVC)

First, you need to build all solution (i.e. train the SVC model) by running the Jupyter Notebook ```_pipeline.ipynb``` available at ```TF/computer_vision/segment_cell```. For visualizing each step of the solution, open the Jupyter Notebook ```source.ipynb``` available in the ```TF/computer_vision``` folder.

### Deep Learning
- This Part of the TF was developed using the Pytorch framework.
- Regarding the CNN model used in this work, we choose a modifying U-NET architecture where we change its backbone to use a VGG model and we have also included a ASPP module at the bottom of the U-Net model to handle multi-scale segmentations tasks.

#### How to run (step-by-step):

First of all, you need to install the NVIDIA Automatic Mixed Precision (Amp).
You can find more information about it and also how to install it in this GitHub repo: https://github.com/NVIDIA/apex

#### Training
To train the model for bacteria segmentation, you have to open the ```train.py``` script and modify the paths according to your environment setup.
Having this first step done, just open the terminal inside the ```TF/deep_learning/src``` folder and type:
```python
python3 train.py
```

#### Inference/Tests
If you just want to test our model using your dataset (considering that your data is related to darkfield microscopy images, see https://www.kaggle.com/longnguyen2306/bacteria-detection-with-darkfield-microscopy for more details), you have to download the pretrained model, available at: https://www.dropbox.com/s/0of9ref6fosit5a/unet_vgg_aspp.pth?dl=0

Otherwise, if you have trained a new model, just follow the steps bellow.
 
After this, you have to open the test or inference script, both available in the ```TF/deep_learning/src``` folder, and again modify the paths according to your environment setup.
Then, just open the terminal (or your IDE) and type:
- For Testing:
```python
python3 test.py
```
- For Inference:
```python
python3 inference.py
```
