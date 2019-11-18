
import sys
import os
import data
import recognition
import GAN
import tensorflow as tf

















#main function
if __name__ =='__main__':
    #run processes
    #email when done
    print('Tensorflow Version: ', tf.__version__)
    directory = sys.argv[1:][0]
    print(directory)
    #data = data.Data(directory[0])
    #train_gen, val_gen = data.train_datagen, data.test_datagen
    #recognition_model = recognition.Recognition()
    #recognition_model.train_and_visualize(train_gen, val_gen)
    data = data.Data(directory).data_array
    joe = GAN.GAN()
    generated_img = joe.test_artist(43)
    joe.test_critic(generated_img)
    joe.train(data, 50)

