import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pickle
from tensorflow.keras.layers import Dense, Dropout,Flatten,Activation,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Sequential


def data(IMG_SIZE):
    DIR = "./"
    CATEGORIES = ['ClosedFace', 'OpenFace']
    train_data = []
    for cat in CATEGORIES:
        path = os.path.join(DIR,cat)
        print(path)
        labels = CATEGORIES.index(cat)
        for img in os.listdir(path):
            try:
                pixel_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                pixel_array = cv2.resize(pixel_array,(IMG_SIZE,IMG_SIZE))
                
                train_data.append([pixel_array, labels])
            except Exception as e:
                pass
    random.shuffle(train_data)
    X, y = [],[]
    
    for img, labels in train_data:
        
        
        X.append(img)
        y.append(labels)
    
    X = np.array(X,'float32')
    X /= 255
    print(X)
    y = tf.keras.utils.to_categorical(y,num_classes=2)
    X= np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    
    pickle_out = open('X.pickle','wb')
    pickle.dump(X,pickle_out)
    pickle_out.close()
    pickle_out = open('y.pickle','wb')
    pickle.dump(y,pickle_out)
    pickle_out.close()
    return (X,y)

def compile_model(input_shape):
    model = Sequential()

    model.add(Conv2D(128, (3, 3), input_shape=input_shape,activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3),activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))

    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model
def test_model(input_shape):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model
    
    
if __name__ == "__main__":
    IMG_SIZE = 48

    if os.path.isfile("X.pickle"):
        print("file found...")
        pickle_in = open("X.pickle",'rb')
        X = pickle.load(pickle_in)  
        
        pickle_in = open("y.pickle",'rb')
        y = pickle.load(pickle_in)  
        
        
    else:
        
        print("processing data../")
        X, y = data(IMG_SIZE)

   
        
    model = compile_model((IMG_SIZE,IMG_SIZE,1))
    

    
    if os.path.isfile('eye.h5'):
        print('model already trained...')
    
    else:
        model.fit(X, y, batch_size=32 ,epochs=10,validation_split=0.3)
        model.save('eye.h5')

