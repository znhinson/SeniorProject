# CHC Model
> This CHC Model  is a system that performs hair classification on images via deep learning and a convolutional neural network (in the form of a modified VGG model).

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Setup](#setup)
* [Room for Improvement](#room-for-improvement)



## General Information
The CHC model is a system that performs multi-class classification on images via deep learning and
a convolutional neural network (in the form of a modified VGG model). The CHC model uses multi-class
classification to identify if an image showcases no hair or one of the ten hair types (i.e., 1, 2A, 2B,
2C, 3A, 3B, 3C, 4A, 4B, 4C). Users can interact with the CHC model via a user interface. Users will upload
an image of their hair to the interact. The user image will be analyzed by the trained CHC model to produce
a hair type prediction. The prediction will then be visualized to the user via the user interface.


## Technologies Used
- Python
- Google Colab
- Visual Studio Code (VS Code)



## Setup
1. Download the [Jupyter Notebook]() file 

2. Upload notebook to an IDE that works with Jupyter Notebook and Gradio (suggestions: VS Code or Google Colab)

3. To install all of the required software dependencies, run the following scripts in IDE:

      **pip install keras**

      **pip install tensorflow**

      **pip install gradio**

4. Download [Hair Dataset]() to local drive

5. In the pre-processing code block, change the DIRECTORY to reflect where the hair dataset is located on local drive

6. Run each block of code to implement the CHC Model



## Room for Improvement
These are areas in which the CHC model could improve.

Room for improvement:
- Address the overfitting issue of the CHC model
- Increase hair dataset




