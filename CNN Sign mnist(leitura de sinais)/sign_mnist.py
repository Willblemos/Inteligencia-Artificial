import time
from time import perf_counter as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

num_classes = 24
batch_size = 125
epochs = 15

train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv') #Rota do dataset, usei pelo próprio notebook da kaggle. Pra rodar localmente tem que mudar isso. 
test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')

#Extraindo dados 
labels = train['label'].values
unique_val = np.array(labels)

#Converte multi-class labels para labels binárias --> pertence ou não pertence a classe 
from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer() 
labels = label_binrizer.fit_transform(labels)


train.drop('label', axis = 1, inplace = True) # pandas.drop seleciona um dataframe de index/column labels sem a parte removida --> separa dados de treino para trabalhar com o resto dos dados
images = train.values
plt.style.use('grayscale') 
images =  images/255 #Os valores variam de 0 a 255, então para normalizar, dividimos cada entrada por 255.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, stratify = labels, random_state = 7)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = Sequential() # Model onde os layers são empilhados sequencialmente. 
#layer com 64 filtros e ativação relu, que tem como entrada 28*28 pixeis da imagem.
model.add(Conv2D(64, kernel_size=(4,4), activation = 'relu', input_shape=(28, 28 ,1), padding='same' ))
model.add(Dropout(0.4)) #Descarta unidades aleatoriamente, ajuda a evitar overfitting 
model.add(MaxPooling2D(pool_size = (2, 2))) #redução de tamanho

model.add(Conv2D(64, kernel_size = (4, 4), activation = 'relu', padding='same' ))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten()) #Nivela a entrada
model.add(Dense(128, activation = 'relu')) #camada NN regularmente conectada densamente.
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer='nadam',
              metrics=['accuracy'])

#Aumentar a imagem durante o model fitting. 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(shear_range = 0.25,
                                   zoom_range = 0.15,
                                   rotation_range = 15,
                                   brightness_range = [0.15, 1.15],
                                   width_shift_range = [-2,-1, 0, +1, +2],
                                   height_shift_range = [ -1, 0, +1],
                                   fill_mode = 'reflect')
test_datagen = ImageDataGenerator()

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size) #Rodando

#Validação dos dados.
test_labels = test['label']
test.drop('label', axis = 1, inplace = True)
test_images = test.values/255
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images.shape

#Acurácia de predição
y_pred = model.predict(test_images)
from sklearn.metrics import accuracy_score
y_pred = y_pred.round()
print("Acuracia media:" +str(accuracy_score(test_labels, y_pred)))
#accuracy_score(test_labels, y_pred)


