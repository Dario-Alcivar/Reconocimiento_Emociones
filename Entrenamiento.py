import collections
import soundfile # lee un archivo de audio
import numpy as np
import librosa # extrae características del audio
import glob 
import os, sys, stat
import pickle # guardar el modelo después del entrenamiento
from sklearn.model_selection import train_test_split # para dividir entrenamientos y pruebas
from sklearn.neural_network import MLPClassifier # modelo de perceptrón multicapa
from sklearn.metrics import accuracy_score # Para medir la precisión
#Para la matriz de confusion
from sklearn.metrics import confusion_matrix
import pandas as pd                         
import seaborn as sns                         
import matplotlib.pyplot as plt

def extract_feature(file_name, mfcc, chroma, mel):   
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma: 
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma_= librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            chroma = np.mean(chroma_.T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
    return result

# todas las emociones en el conjunto de datos de RAVDESS
int2emotion = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}

# permitimos solo estas emociones 
AVAILABLE_EMOTIONS = [
    'calm', 
    'happy', 
    'fearful', 
    'disgust'
]


def load_data(test_size=0.2):
    X, y = [], []
    canttotalaprocesar=len(glob.glob("DataFlair\\Actor_*\\*.wav"))
    cont=0
    for file in glob.glob("DataFlair\\Actor_*\\*.wav"):
        # obtener el nombre base del archivo de audio
        basename = os.path.basename(file)
        # obtener la etiqueta de emoción
        emotion = int2emotion[basename.split("-")[2]]
        # permitimos las EMOCIONES DISPONIBLES 
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extraer características del audio
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # agregar a los datos
        X.append(features)
        y.append(emotion)

        cont=cont+1
        porcentaje=(cont/canttotalaprocesar)/100
        print("Elementos procesados: "+ str(cont)+"/"+ str(canttotalaprocesar) + "  "+ str(porcentaje)+"%")
    # dividir los datos para entrenamiento y prueba y devolverlos
    return train_test_split(np.array(X), y, test_size=test_size, random_state=9)

##prueba
#prueba='DataFlair\\Actor_16\\03-02-03-01-01-01-16.wav'
#basename = os.path.basename(prueba)
#emocion=int2emotion[basename.split("-")[2]]

#features_ = extract_feature(prueba, mfcc=True, chroma=True, mel=True)        
#print("Tu estado de ánimo es: ",emocion)

## cargar datos RAVDESS , 75% entrenamiento 25% prueba
X_train, X_test, y_train, y_test = load_data(test_size=0.25)

## imprimir algunos detalles
## número de muestras en los datos de entrenamiento
print("[+] Number of training samples:", X_train.shape[0], X_test.shape[0])
## numeros de características usadas
## este es un vector de características extraídas usando la función extract_features ()
print("[+] Number of features:", X_train.shape[1])

# mejor modelo, determinado por una búsqueda de cuadrícula
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}

 # inicializar el clasificador de perceptrón multicapa
model = MLPClassifier(**model_params) 

## entrenar el modelo
print("[*] Training the model...")
model.fit(X_train, y_train)

# predecir el 25% de los datos para medir la precisión
y_pred = model.predict(X_test)

# calcular la precisión
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
##imprimir la precision
print(y_pred)
#print(f'Tu estado de animo es: ',features_)

## Ahora guardamos la modelo
## hacer directorio de resultados si aún no existe
#if not os.path.isdir("result"):
#    os.mkdir("result")
     
#pickle.dump(model, open("result/mlp_classifier.model", "wb"))

#resultado temp de presición "92.75%"
