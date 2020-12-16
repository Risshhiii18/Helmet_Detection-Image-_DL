import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils


DATADIR = "D:/NEww/Data/training"
CATEGORIES = ['with helmet','without helmet']

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img_array, cmap='gray')
        #plt.show()
        
print(img_array)
print(img_array.shape)


IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in (os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
            
create_training_data()

print(len(training_data))

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_train = X/255.0
number_of_classes = 2
y_train = np_utils.to_categorical(y,number_of_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.summary()

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, y, validation_data=(X_train, y), epochs=10, batch_size=50)

model.save('D:/l1p29984.model')

#############################################################################################

test_data = []
DAT = "D:/NEww/Data"
CAT = ['test']

def new_data():
    for category in CAT:

        path = os.path.join(DAT,category)

        for img in (os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            test_data.append([new_array, img])

new_data()

Xt = []
yt =[]

for features,lab in test_data:
    Xt.append(features)
    yt.append(lab)
    
Xt = np.array(Xt).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

Xt = Xt/255.0
pred = model.predict(Xt)
import pandas as pd
predict = pd.DataFrame(pred, index=yt)
plt.plot(predict)
predict[0] = np.where(predict[0] > 0.73, 2,predict[0])
predict[0] = np.where(predict[0] < 0.73, 1,predict[0])
predict[0] = np.where(predict[0] == 2, 0,predict[0])
predict[0] = predict.astype(int)
predict.to_csv('D:/Attempt2helmet.csv')
##############################################################################################