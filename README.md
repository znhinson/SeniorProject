# CHC Model
> The CHC model is a system that performs hair classification on images via deep learning and a convolutional neural network (VGG model).

## Table of Contents
* [General Info](#general-information)
* [Required Software Dependencies](#required-software-dependencies)
* [Setup](#setup)
* [Getting Started](#getting-started)



## General Information
The CHC model is a system that performs multi-class classification on images via deep learning and
a convolutional neural network (in the form of a VGG model). The CHC Mmodel uses multi-class classification
to identify if an image has no hair or one of the ten hair types (i.e., 1, 2A, 2B, 2C, 3A, 3B, 3C, 4A, 4B, 4C).
A user interface has been developed to allow users to upload an image of their hair. The user image
will be tested against the trained CHC model, producing a hair type prediction. The result from the trained
CHC model is visualized to the user via the user interface.


## Required Software Dependencies
- Python (3.7 version or later)
- Visual Studio (VS) Code




## Setup

### Installations
Run the following in terminal:


![Keras](https://github.com/znhinson/SeniorProject/blob/main/images/install_1.PNG)


![Tensorflow](https://github.com/znhinson/SeniorProject/blob/main/images/install_2.PNG)


![Gradio](https://github.com/znhinson/SeniorProject/blob/main/images/install_3.PNG)

 

### Downloading Files
1. Click **Code**
2. Click **Download ZIP**
3. Click **Extract all** for the **SeniorProject-main.zip** folder

## Getting Started

### Setting up environment in VS Code
1. Launch VS Code 
2. Click New Folder
3. Select **SeniorProject-main** folder
4. Navigate to *Explorer*
5. Select **v2_base** folder
6. Click **vgg_model.ipynb** file 


After the completion of these steps, run each block of code to implement the CHC Model


## User Tutorial

### How to Upload an Image
1. Upload hair image (JPG/JPEG or PNG) to the left window
2. Click **Submit**
3. View hair type prediction results in the right window

### How to clear submission
1. Click **Clear**

## Room for Improvement
These are areas in which the CHC Model could improve.

- Address the overfitting issue of the CHC Model
- Increase hair dataset




