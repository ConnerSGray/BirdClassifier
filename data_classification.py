"""

Conner Gray - data_training.py - 3/16/18

Use the trained model to classify new pictures of 
bird species collected by the PASSER smart feeder

"""


import cv2
import numpy as np
import tensorflow as tf
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TEST_DIR = 'data_dir/validation'
MODEL_NAME = 'bird-{}.model'.format('V1-3/28')

#Make the test data so that we have
def mk_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        trash, img_num = img.split('(')
        img_num, trash = img_num.split(')')
        img = cv2.imread(path)
        img = cv2.resize(img, (320, 240))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
        
#Versions of the neural network need to be in both the trainer and 
#classifier so that they can both hold the same architecture
convnet = input_data(shape=[None, 320, 240, 3], name='input')

convnet = conv_2d(convnet, 64, 9, activation='relu')
convnet = max_pool_2d(convnet, 9)

convnet = conv_2d(convnet, 32, 9, activation='relu')
convnet = max_pool_2d(convnet, 9)

convnet = conv_2d(convnet, 64, 9, activation='relu')
convnet = max_pool_2d(convnet, 9)


convnet = conv_2d(convnet, 32, 9, activation='relu')
convnet = max_pool_2d(convnet, 9)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.6)

convnet = fully_connected(convnet, 9, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded')


if os.path.exists('test_data.npy'):
    test_data = np.load('test_data.npy')
    shuffle(test_data);
else:
    test_data = mk_test_data()


#Currently classified in a grid of 16 images as a placeholder 
#until the data is pushed to its intended location
figure = plt.figure(figsize=(10,9))

for num, data in enumerate(test_data[32:48]):
    
    img_num = data[1]
    img_data = data[0]
    
    y = figure.add_subplot(4,4,num+1)
    bgr = img_data
    rgb = bgr[...,::-1]
    data = img_data.reshape(320,240,3)
    
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 0: str_label='Female Cardinal'
    elif np.argmax(model_out) == 1: str_label='Male Cardinal'
    elif np.argmax(model_out) == 2: str_label='American Goldfinch'    
    elif np.argmax(model_out) == 3: str_label='Black-capped Chickadee'
    elif np.argmax(model_out) == 4: str_label='Blue Jay'
    elif np.argmax(model_out) == 5: str_label='Eastern Tufted Titmouse'
    elif np.argmax(model_out) == 6: str_label='Downy Woodpecker'
    elif np.argmax(model_out) == 7: str_label='Male House Finch'
    elif np.argmax(model_out) == 8: str_label='Nothing'
        
        
    y.imshow(rgb)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.show()

#Make sure we find a way to see that there is not an identified 