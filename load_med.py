import os
import cv2
import numpy as np

from keras import backend as K
from keras.utils import np_utils

from data import DataLoader

NB_TRAIN_SAMPLES = 3000 # 3000 training samples
NB_VALID_SAMPLES = 100 # 100 validation samples
NUM_CLASSES = 23
TRAIN_FOLDER = os.path.abspath('./med_data/train_emb')
TEST_FOLDER = os.path.abspath('./med_data/test_emb')

def load_med_data(img_rows, img_cols):
    # Load our training and validation sets
    train_loader = DataLoader(TRAIN_FOLDER)
    test_loader = DataLoader(TEST_FOLDER)
    (X_train, Y_train) = train_loader.load_data()
    (X_valid, Y_valid) = test_loader.load_data()

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:NB_TRAIN_SAMPLES,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:NB_VALID_SAMPLES,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:NB_TRAIN_SAMPLES]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:NB_VALID_SAMPLES]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:NB_TRAIN_SAMPLES], NUM_CLASSES)
    Y_valid = np_utils.to_categorical(Y_valid[:NB_VALID_SAMPLES], NUM_CLASSES)

    return X_train, Y_train, X_valid, Y_valid