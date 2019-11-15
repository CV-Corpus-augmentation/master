import tensorflow as tf
import numpy as np
import sys
import os


from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

class Data(object):
    def __init__(self, directory):
        self.target_size = (80, 80)
        self.batch_size = 16
        self.train_datagen = self.build_train_datagen(directory)
        self.test_datagen = self.build_test_datagen(directory)
        self.data_array = self.build_array()



    def build_train_datagen(self, directory):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        train_gen = train_datagen.flow_from_directory(directory + '/training',
                                                      target_size=self.target_size, color_mode='grayscale',
                                                      batch_size=self.batch_size,
                                                      class_mode='binary')
        return train_gen

    def build_test_datagen(self, directory):
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        val_gen = test_datagen.flow_from_directory(directory + '/val',
                                                   target_size=self.target_size, color_mode='grayscale',
                                                   batch_size=self.batch_size,
                                                   class_mode='binary')

        return val_gen


    def build_array(self):
        dataset = np.array(next(self.train_datagen)[0])
        for x in range(12):
            t = np.array(next(self.train_datagen)[0])
            dataset = np.append(dataset, t, axis=0)

        ds2 = dataset[:192].reshape((12, 16, 80, 80, 1))
        return ds2

