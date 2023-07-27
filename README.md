# Lip-Read-ML-Model

This is a simple machine learning model which can take a video of a person speaking and predict what it was.

# Overview

1. Used tensorflow for building the model
2. Used keras for data processing and numpy for better array usage
3. Used sequential model for training and prediction
4. Used relu as the activation
5. I have used 3 conv3d, 2 bidirectional lstm layers for traning
6. Used adam optimizer and ctc loss for training
7. Used imageio for reading the video and cv2 for getting the frames

# Basic Logic

1. Made two functions load_video and load_alignments for loading the video and the alignments
2. Video function convert the video frames to grayscale and crops to the mouth portions for lesser training time
3. The alignments function use the word outputs from the files and store it as tokens later convert them to numbers
4. Now we use a mappable function to get all the inputs to the function.
5. we build a sequential model with 3 conv3d layers and 2 bidirectional lstm layers using tensorflow keras layers
6. We use relu as the activation function and adam as the optimizer
7. We use ctc loss for training the model
8. we train the model using the fit function for particular epochs (over 90 epochs for better accuracy)
9. We use the model to predict the speech of the video

# How to run

1. Download the dataset from the link given below
2. Extract the dataset and place it in the same folder as the code
3. Run the code using jupyter notebook or any other IDE
4. The code will train the model and predict the speech of the video

# Code

model layers used in the code

```python
model = Sequential()
# conv3D used for video processing input shape is the shape of each frame and 128 output filters and 3 is 3d kernel size
model.add(Conv3D(128,3,input_shape=(75,46,140,1),padding='same'))
# to get some non linearities
model.add(Activation('relu'))
# takes max values of each frame and condences into 2x2 kernel
model.add(MaxPool3D((1,2,2)))

# 2nd layer with 256 output filters
model.add(Conv3D(256,3,padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75,3,padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

# flatten the output to feed into dense layer
model.add(TimeDistributed(Flatten()))

# 2 layer LSTM
# return_sequences=True means it will return the output of each time step
# dropout to prevent overfitting
# kernel_initializer='Orthogonal' to prevent vanishing gradient problem
# 128 is the number of hidden units
model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal',return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal',return_sequences=True)))
model.add(Dropout(.5))

# dense layer with softmax activation
# output in the form of one hot encoding of the characters in the vocabulary + 1 for blank character
# using softmax activation to get the probability of each character then take the max probability using argmax
model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal',activation='softmax'))
```

# basic imports

```python
import gdown
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPool3D, TimeDistributed, LSTM, Bidirectional
import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
```

# DataSet

1. [https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL](https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL)

# Basic commands

1. `pip install tensorflow`
2. `pip install keras`
3. `pip install numpy`
4. `pip install imageio`
5. `pip install matplotlib`
6. `pip install cv2`

# References

1. [Youtube](https://www.youtube.com/watch?v=uKyojQjbx4c)
