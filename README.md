# CHC Model
> This CHC Model  is a system that performs hair classification on images via deep learning and a convolutional neural network (VGG Model).

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Setup](#setup)
* [Room for Improvement](#room-for-improvement)



## General Information
The CHC model is a system that performs multi-class classification on images via deep learning and
a convolutional neural network (VGG Model). The CHC Model uses multi-class classification to identify if an
image has no hair or one of the ten hair types (i.e., 1, 2A, 2B, 2C, 3A, 3B, 3C, 4A, 4B, 4C). A user interface
was developed to allow users to upload an image of their hair. The user image will be tested against the trained 
CHC Model, producing a hair type prediction. The results of the trained CHC Model is visualized to the user via 
user interface.


## Technologies Used
- Python
- Google Colab



## Setup
1. Download the [Jupyter Notebook]() file 

2. Upload notebook to Google Colab

3. To install all of the required software dependencies, run the following scripts in Google Colab:

      **pip install keras**

      **pip install tensorflow**

      **pip install gradio**

4. Download [Hair Dataset]() 

5. Save the datset to your Google Drive 

6. To access the hair dataset via Google Colab, Mount Google Drive 

7.In the pre-processing code block, change the DIRECTORY to reflect where the hair dataset is located in  Google Drive:
![image]()


After completing the following steps, run each block of code to implement the CHC Model



## Room for Improvement
These are areas in which the CHC Model could improve.

Room for improvement:
- Address the overfitting issue of the CHC Model
- Increase hair dataset




