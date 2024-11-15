import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from keras.layers import Input, Dense, LSTM, concatenate, Activation, GRU, SimpleRNN
from keras.models import Model

def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=64):
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = LSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=128, normalize_timeseries=False):
    if normalize_timeseries:
        X_train_mean = X_train.mean()
        X_train_std = X_train.std()
        X_train = (X_train - X_train_mean) / X_train_std
        X_test = (X_test - X_train_mean) / X_train.std()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history

def evaluate_model(model, X_test, y_test, batch_size=128, normalize_timeseries=False):
    if normalize_timeseries:
        X_test_mean = X_test.mean()
        X_test_std = X_test.std()
        X_test = (X_test - X_test_mean) / X_test.std()

    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    return accuracy

def plot_training_history(history):
    # Grafico della loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('model_loss.png')
    plt.close()

    # Grafico dell'accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.savefig('model_accuracy.png')
    plt.close()

# Carica il tuo dataset
data = pd.read_csv('/home/leonardo/Desktop/LSTM-FCN-master/resistance_values.csv')

# Supponiamo che l'ultima colonna sia l'etichetta
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalizza i dati
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Converti le etichette in one-hot encoding
y = to_categorical(y)

# Dividi i dati in training e test set in maniera casuale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aggiungi una dimensione per adattarsi alla forma di input del modello
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Definisci i parametri del modello
MAX_SEQUENCE_LENGTH = X_train.shape[2]
NB_CLASS = y_train.shape[1]
NUM_CELLS = 64  # Puoi cambiare questo valore

# Libera la memoria GPU
K.clear_session()

# Genera il modello
model = generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS)

# Addestra il modello
history = train_model(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=128, normalize_timeseries=False)

# Valuta il modello
accuracy = evaluate_model(model, X_test, y_test, batch_size=128, normalize_timeseries=False)

print(f'Test accuracy: {accuracy}')

# Genera e salva i grafici
plot_training_history(history)