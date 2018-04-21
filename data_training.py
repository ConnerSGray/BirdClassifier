"""

Conner Gray - data_training.py - 3/16/18

Using Tensorflow libraries, we attempt to classify different
bird species images collected by the PASSER smart feeder

"""




import cv2
import numpy as np
import tensorflow as tf
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = 'data_dir/training'
TEST_DIR = 'data_dir/validation'
LR = 1e-3

MODEL_NAME = 'bird-{}.model'.format('V1-3/28')

#Assign a label to each image based on the name of the image being parsed
def label_img(img):
    label_name, trash = img.split('(')
    if label_name == 'fnoca ': return [1,0,0,0,0,0,0,0,0]     #Female Cardinal
    elif label_name == 'mnoca ': return [0,1,0,0,0,0,0,0,0]   #Male Cardinal
    elif label_name == 'amgo ': return [0,0,1,0,0,0,0,0,0]    #American Goldfinch
    elif label_name == 'bcch ': return [0,0,0,1,0,0,0,0,0]    #Black-capped Chickadee
    elif label_name == 'blja ': return [0,0,0,0,1,0,0,0,0]    #Blue Jay
    elif label_name == 'etti ': return [0,0,0,0,0,1,0,0,0]    #Eastern Tufted Titmouse
    elif label_name == 'fdowo ': return [0,0,0,0,0,0,1,0,0]   #Female Downy Woodpecker
    elif label_name == 'mhofi ': return [0,0,0,0,0,0,0,1,0]   #Male House Finch
    elif label_name == 'nothing ': return [0,0,0,0,0,0,0,0,1] #Nothing pictured

#Pull in the images from the training directory and format them for
#training
def mk_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path)
        img = cv2.resize(img, (320, 240))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
        
train_data = mk_train_data()


#The neural network consists of 4 layers of ReLU activation functions,
#a large "Fully Connected" layer that is not truly fully connected but
#still serves the same purpose, and a final layer that spits out the
#probabilities of each
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

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

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, 320,240, 3)
Y = [i[1] for i in train]
Y = np.reshape(Y, (-1, 9))

test_X = np.array([i[0] for i in test]).reshape(-1, 320,240, 3)
test_Y = [i[1] for i in test]

#Start training the data
model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_X}, {'targets': test_Y}), 
    snapshot_step=50, show_metric=True, run_id=MODEL_NAME)

#tensorboard --logdir=logginginfo:"C:\Users\condo\Desktop\Coding Workspace\Python\BirdClassifier\log"
model.save(MODEL_NAME)



