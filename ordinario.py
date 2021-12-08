#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 22:38:07 2021

@author: pcc
"""
import pandas as pd
import seaborn as sns
import speech_recognition as sr
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile 
import scipy.io.wavfile as waves
import numpy as np
#________
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import MLPClassifier
from sklearn.neural_network import MLPClassifier
# INGRESO
archivo='GATO.wav'
fs, sonido = waves.read(archivo)

print('Frecuencia de Muestreo = ',fs)
n = len(archivo) 
Audiodata = sonido / (2.**15) # # definir amplitud de los datos entre [-1 : 1]
AudioFreq = fft(Audiodata) # Calcular la transformada de Fourier
# La salida de la FFT es un array de numeros complejos
MagFreq = np.abs(AudioFreq) # Valor absoluto para obtener la magnitud
# dependan del tamaño de la señal o de su frecuencia de muestreo
#MagFreq = MagFreq / float(n)
#print(MagFreq)

# Calcular el espectro de potencia
#MagFreq = MagFreq**2
#plt.plot(10*np.log10(MagFreq)) #Espectro de potencia
#plt.plot(MagFreq) #Espectro de magnitud
#plt.show()
#arreglo con magnitudes de frecuencia
#prueba 4 para clasificar audio con un xor
xs = np.array([
    MagFreq[1], 0,
    0, 1,
    1, 0,
    1, 1
]).reshape(4, 2)
print(xs)#ver el arreglo

#arrayys
ys = np.array([0, 1, 1,0]).reshape(4,)

#Se crea el clasificador con la función de activación relu,con 10k iteraciones y tiene capaz ocultas 4,2
model = MLPClassifier(activation='tanh', max_iter=100, hidden_layer_sizes=(4,2))
#Se entrena la red neuronal pasando los arreglos de entrada y de salida
model.fit(xs, ys)
print('prediccion:', model.predict(xs))
#prueba clasificador pasando datos de obtencion de frecuencia de muestreo y magnitud de frecuencia 
#en un csv coloque los datos de obtencion de los audios
#lectura del csv
datosgp = pd.read_csv("datos.csv")
#como el id no es relevante se quita
#Eliminamos la primera columna ID
datosgp = datosgp.drop('id',axis=1)
print('perros ygatos:')
print(datosgp.groupby('quees').size())
#Grafico Sepal - Longitud vs Ancho
fig = datosgp[datosgp.quees == 'gato'].plot(kind='scatter', x='freq', y='magfreq', color='blue', label='Gato')
datosgp[datosgp.quees == 'perro'].plot(kind='scatter', x='freq', y='magfreq', color='green', label='Perro', ax=fig)


#clasificacion con su caracteristica (gato/perro)
X = np.array(datosgp.drop(['quees'], 1))
y = np.array(datosgp['quees'])
#entrenamiento y pruebas(porcentaje)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Modelo de Regresión Logística
"""
definir el algoritmo, LogisticRegression, seguidamente lo entrenamos utilizando la instruccion fit 
y realizamos la una prediccion utilizando los datos de X_test. 
Para determinar la precision o confianza del algoritmo utilizamos la instruccion score para calcularla.
"""
algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)

#datos con las características y las etiquetas o resultados
frecuencias = datosgp[['freq','magfreq','quees']]
X_f = np.array(frecuencias.drop(['quees'], 1))
y_f = np.array(frecuencias['quees'])
# separar los datos de entrenamiento y prueba para proceder a construir los modelos.

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_f, y_f, test_size=0.2)

print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))
#Modelo de Regresión Logística
algoritmo = LogisticRegression()
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisión Regresión Logística : {}'.format(algoritmo.score(X_train_s, y_train_s)))
fig.set_xlabel('Frecuencia')
fig.set_ylabel('Magnitud')

plt.show()