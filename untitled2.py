import tensorflow as tf
from multiprocessing.dummy import active_children
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns 

cifar10 = tf.keras.datasets.cifar10

(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()

X_treinamento = X_treinamento.reshape((len(X_treinamento), np.prod(X_treinamento.shape[1:])))
X_teste = X_teste.reshape((len(X_teste), np.prod(X_teste.shape[1:])))

X_treinamento = X_treinamento.astype('float32')
X_teste = X_teste.astype('float32')

X_treinamento /= 255
X_teste /= 255

y_treinamento = np_utils.to_categorical(y_treinamento, 10)
y_teste = np_utils.to_categorical(y_teste, 10)

modelo = Sequential()
modelo.add(Dense(units = 256, activation = 'relu', input_dim = 3072))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 128, activation = 'relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 64, activation = 'relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 10, activation = 'softmax'))

modelo.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

previsoes = modelo.predict(X_teste)
y_teste_matriz = [np.argmax(a) for a in y_teste]
y_previsoes_matriz = [np.argmax(a) for a in previsoes]
confusao = confusion_matrix(y_teste_matriz, y_previsoes_matriz)
confusao_df = pd.DataFrame(confusao)

confusao_df
