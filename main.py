
import sys
import os
import data
import recognition
















#main function
if __name__ =='__main__':
    #run processes
    #email when done
    print('Tensorflow Version: ', tf.__version__)
    directory = sys.argv[1:][0]
    print(directory[0])
    data = data.Data(directory[0])
    train_gen, val_gen = data.train_datagen, data.test_datagen
    recognition_model = recognition.Recognition()
    recognition_model.train_and_visualize(train_gen, val_gen)

