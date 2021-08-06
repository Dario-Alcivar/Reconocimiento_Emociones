import collections
import soundfile # para leer un archivo de audio 
import numpy as np
import librosa # para extraer características del audio
import telebot  #utilizar el bot de telegram
import subprocess  #realizar subprocesos
import glob 
import os, sys, stat
import pickle # guardar el modelo después del entrenamiento
import os              
import tensorflow as tf
import pandas as pd                         
import seaborn as sns                         
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # para dividir entrenamientos y pruebas
from sklearn.neural_network import MLPClassifier # modelo de perceptrón multicapa
from sklearn.metrics import accuracy_score # Para medir la precisión                    
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout 
from sklearn.metrics import confusion_matrix
from datetime import datetime 

# importar de Entrenamiento 
import Entrenamiento as e

# Ingresar el token reeplazanto TOKEN. 
bot = telebot.TeleBot('TOKEN')
@bot.message_handler(content_types = ['voice', 'audio'])
 

def get_audio_messages(message):      
    audioentradas='audioentradas\\'
    audioconvertidos='audioconvertidos\\'

    file_info = bot.get_file(message.voice.file_id)
    #descarga el archivo audio
    downloaded_file = bot.download_file(file_info.file_path)
    #crea e archivo en la ruta base con un nombre especifico
    nombrearchivosinextension=datetime.today().strftime('%Y-%m-%d %H%M%S%f')
    nombrearchivo =nombrearchivosinextension +'.ogg'
    with open(audioentradas+nombrearchivo, 'wb') as new_file:
        new_file.write(downloaded_file)

    src_filename = audioentradas+nombrearchivo
    dest_filename = audioconvertidos+nombrearchivosinextension+'.wav'

    process = subprocess.run(['ffmpeg\\bin\\ffmpeg.exe', '-i', src_filename, dest_filename])

     # Extraer características
    msgsentimiento=ProcesarIAReconocimientoSentimientos(dest_filename)
    
     # Extraer características
    text = 'Inicio el proceso de reconocimiento de emoción, por favor espere unos minutos. Gracias :)'
    bot.send_message(message.chat.id, text)   
    ty = e.Bot()
    ob = np.load('prueba.npy')
    str = ' '.join(ob)
    text = 'Tu emocion correponde al siguiente dígito' + ' ' + str 
    bot.send_message(message.chat.id, text) 
    text = 'Posición 1 = Enojo , 2 = Tranquilo , 3 = Temeroso , 4 = Feliz , 5 = Neutral , 6 =triste'  
    bot.send_message(message.chat.id, text)



def ProcesarIAReconocimientoSentimientos(urlWAV):      
    print ("\nRecolectando características...")   
    print("\nEste proceso tomará un tiempo...")                        
    features = parse_audio_files(urlWAV)                      
    print("Finalizado")                         
    np.save('X2', features)   
    X = np.load('X2.npy', allow_pickle=True) 

def extract_feature(file_name): 
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc=40).T, axis = 0)
    chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr=sample_rate).T, axis = 0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr = sample_rate).T, axis = 0)
    contrast = np.mean(librosa.feature.spectral_contrast(S = stft, sr = sample_rate).T, axis = 0)
    tonnetz = np.mean(librosa.feature.tonnetz(y = librosa.effects.harmonic(X),
    sr = sample_rate).T,axis = 0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(urlWAV):
    features, labels = np.empty((0,193)), np.empty(0)
    try:
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(urlWAV)
    except Exception as e:
        print ("Error encountered while parsing file: ", urlWAV)
    
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = np.vstack([features, ext_features])
    return np.array(features)

bot.polling(none_stop = True)