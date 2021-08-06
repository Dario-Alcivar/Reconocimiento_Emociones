class Bot:
   def __init__(self):
        import glob                         
        import os                         
        import librosa                         
        import numpy as np   
        import tensorflow as tf
        from tensorflow import keras as keras
        from tensorflow.keras import layers 
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Dropout              
        from sklearn.model_selection import train_test_split    
        from sklearn.metrics import accuracy_score # Para medir la precisión
        from sklearn.metrics import confusion_matrix
        import pandas as pd                         
        import seaborn as sns                         
        import matplotlib.pyplot as plt
        import main as main

        def extract_feature(file_name): 
            X, sample_rate = librosa.load(file_name)
            stft = np.abs(librosa.stft(X))
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
            sr=sample_rate).T,axis=0)
            return mfccs,chroma,mel,contrast,tonnetz

        def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
            features, labels = np.empty((0,193)), np.empty(0)
            for label, sub_dir in enumerate(sub_dirs):
                for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                    try:
                        mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
                    except Exception as e:
                        print ("Error encountered while parsing file: ", fn)
                    continue
                    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                    features = np.vstack([features,ext_features])
                    labels = np.append(labels, fn.split('\\')[2].split('-')[2])
            return np.array(features), np.array(labels, dtype = np.int)

        def one_hot_encode(labels):                           
            n_labels = len(labels)                           
            n_unique_labels = len(np.unique(labels))
            one_hot_encode = np.zeros((n_labels,n_unique_labels+1))                           
            one_hot_encode[np.arange(n_labels), labels] = 1                           
            one_hot_encode=np.delete(one_hot_encode, 0, axis=1)                           
            return one_hot_encode

        main_dir = 'DataFlair'                       
        sub_dir=os.listdir(main_dir)                        
        print ("\ncollecting features and labels...")   
        print("\nthis will take some time...")                        
        features, labels = parse_audio_files(main_dir,sub_dir)                      
        print("done")                         
        np.save('X',features)                         
        labels1 = one_hot_encode(labels)  
        np.save('X1',labels1)                                           
        X=np.load('X.npy')  
        Y=np.load('X1.npy') 
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.33, random_state=42)


        n_dim = train_x.shape[1]                        
        n_classes = train_y.shape[1]                         
        n_hidden_units_1 = n_dim                         
        n_hidden_units_2 = 400                        
        n_hidden_units_3 = 200 
        n_hidden_units_4 = 100            

                                   
        #Definir el modelo                     
        def create_model(activation_function = 'relu', optimiser ='adam', dropout_rate=0.2):       
            model = keras.Sequential()                       
            # Capa 1      
            model.add(Dense(n_hidden_units_1, input_dim = n_dim, activation = activation_function))                           
            # Capa 2
            model.add(Dense(n_hidden_units_2, activation = activation_function))                           
            model.add(Dropout(dropout_rate))                           
            # Capa 3
            model.add(Dense(n_hidden_units_3, activation = activation_function))                           
            model.add(Dropout(dropout_rate))                           
            # Capa 4                          
            model.add(Dense(n_hidden_units_4, activation = activation_function))                          
            model.add(Dropout(dropout_rate))                          
            # Capa de salida                          
            model.add(Dense(n_classes, activation = 'softmax'))                           
            # Compilación del Modelo                         
            model.compile(loss='categorical_crossentropy', optimizer = optimiser, metrics = ['accuracy'])                          
            return model

        model = create_model()                         
        # Entrenar el modelo                        
        model.fit(train_x, train_y, epochs=200, batch_size=4)

       
        test_y1 = np.argmax(test_y,axis=1)
        predict = np.argmax(model.predict(test_x),axis=1)
        accuracy = accuracy_score(y_true = test_y1, y_pred = predict)
        print("Accuracy: {:.2f}%".format(accuracy * 100))

        #matriz de confusión
        emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']                                                
        predicted_emo = []
        for i in range(0,test_y1.shape[0]):                          
            emo = emotions[predict[i]]  
            predicted_emo.append(emo)

        actual_emo = [] 
        for i in range(0,test_y1.shape[0]): 
            emo = emotions[test_y1[i]] 
            actual_emo.append(emo)

        cm = confusion_matrix(actual_emo, predicted_emo)        
        np.save('data', cm)
        cm2 = np.load('data.npy')
        index = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']                          
        columns = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad'] 
        cm_df = pd.DataFrame(cm2, index, columns)                                           
        #figura = plt.figure(figsize = (6,6))       
        print(cm_df)

        # Probar el modelo
        X = np.load('X2.npy', allow_pickle=True)
        prueba = model.predict(X)
        print("La matriz de emoción es: " ,prueba)    
        np.save('prueba', prueba)

