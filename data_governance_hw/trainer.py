import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import cv2
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (confusion_matrix, 
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
class Trainer():
    def __init__(self, data_dir='./data', BATCH_SIZE=64, IMG_SHAPE=(150,150), EPOCHS=1):
        self.BATCH_SIZE = BATCH_SIZE
        self.IMG_SHAPE  = IMG_SHAPE 
        self.EPOCHS = EPOCHS
        self.data_dir = data_dir
        

    def init_dataset(self):
        train_dir = os.path.join(self.data_dir, 'train')
        validation_dir = os.path.join(self.data_dir, 'test')

        image_gen_train = ImageDataGenerator(rescale=1./255)
        self.train_data_gen = image_gen_train.flow_from_directory(
            batch_size=self.BATCH_SIZE,
            directory=train_dir,
            shuffle=True,
            target_size=self.IMG_SHAPE
            )
        image_gen_val = ImageDataGenerator(rescale=1./255)

        self.val_data_gen = image_gen_val.flow_from_directory(
            batch_size=self.BATCH_SIZE,
            directory=validation_dir,
            shuffle=False,    
            target_size=self.IMG_SHAPE
            )

    def init_model(self):
        self.model = Sequential([
                        Conv2D(32, (3,3), activation='relu', 
                        input_shape=(self.IMG_SHAPE[0], self.IMG_SHAPE[1], 3)),
                        MaxPooling2D(2, 2),
                        Dropout(0.2),
                        Conv2D(64, (3,3), activation='relu'),
                        MaxPooling2D(2,2),
                        Dropout(0.2),
                        Conv2D(128, (3,3), activation='relu'),
                        MaxPooling2D(2,2),
                        Dropout(0.2),
                        Conv2D(256, (3,3), activation='relu'),
                        MaxPooling2D(2,2),
                        Flatten(),
                        Dense(512, activation='relu'),
                        Dropout(0.5),
                        Dense(4, activation='softmax')
                        ])
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['AUC'])
        
        self.checkpoint_filepath = self.data_dir+'/tmp/checkpoint'
        callbacks=[
                EarlyStopping(
                    monitor='val_loss', 
                    patience=4),
                    
                ModelCheckpoint(
                    self.checkpoint_filepath,
                    monitor='val_loss', 
                    save_best_only=True, 
                    save_weights_only=True,
                    mode='auto',
                    verbose=0)
                ]

        self.model.fit(
            self.train_data_gen,
            steps_per_epoch=len(self.train_data_gen),
            epochs=self.EPOCHS,
            validation_data=self.val_data_gen,
            validation_steps=len(self.val_data_gen),
            verbose = 1,
            callbacks=callbacks
        )
        
        

    def evaluate_model(self):
        self.model.load_weights(self.checkpoint_filepath)
        predictions = self.model.predict(self.val_data_gen)
        self.predictions = np.argmax(predictions, axis=1)
        _, AUC = self.model.evaluate(x=self.val_data_gen, batch_size=len(self.val_data_gen), verbose=0)
        print(f'AUC = {np.round((AUC * 100.0), 3)}%')

    def launch_all(self):
        self.init_dataset()
        print()
        print('data was initialized')
        print()
        print()
        self.init_model()
        print()
        print('model was initialized')
        print()
        print()
        self.evaluate_model()
        print()
        print('model was evaluated')
        print()
        print()



