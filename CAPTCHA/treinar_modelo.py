import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

dados = []
rotulos = []
pasta_base_imagens = "base_letras"

imagens = paths.list_images(pasta_base_imagens)

for arquivo in imagens:
    rotulo = arquivo.split(os.path.sep)[-2]
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Padronizando a imagem em 20x20 pixels

    imagem = resize_to_fit(imagem, 20, 20)

    # Adicionando mais uma dimensão para ser compatível com o keras

    imagem = np.expand_dims(imagem, axis=2)

    # Adicionar a imagem e rótulo as listas de dados e rótulos

    rotulos.append(rotulo)
    dados.append(imagem)

dados = np.array(dados, dtype="float") / 255
rotulos = np.array(rotulos)

# Separando em dados de treino (75%) e dados de teste (25%)

(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)

# Convertendo com one-hot encoding

lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Salvando o labelbinarizer em um arquivo com o pickle

with open('rotulos_modelo.dat', 'wb') as arquivo_pickle:
    pickle.dump(lb, arquivo_pickle)

# Criar e treinar a IA

modelo = Sequential()

# Criar as camadas da rede neural

modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Criar a segunda camada

modelo.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Criar a terceira camada

modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))

# Criar a camada de saída

modelo.add(Dense(26, activation="softmax"))

# Compilando todas as camadas

modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinando a IA

modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=26, epochs=10, verbose=1)

# Salvar o modelo em um arquivo

modelo.save("modelo_treinado.hdf5")