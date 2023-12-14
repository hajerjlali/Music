# svm_service.py
from flask import Flask, request, jsonify
import base64
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import librosa
import os

app = Flask(__name__)

# Récupérer le chemin du répertoire courant
current_directory = os.path.dirname(os.path.realpath(__file__))

# Charger les données GTZAN 
data_dir = '/app/data/GTZAN/genres_original'

genres = os.listdir(data_dir)

# Caractéristiques MFCC
X = []
y = []

for genre in genres:
    genre_dir = os.path.join(data_dir, genre)
    for filename in os.listdir(genre_dir):
        file_path = os.path.join(genre_dir, filename)
        audio, _ = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=_, n_mfcc=13), axis=1)
        X.append(mfccs)
        y.append(genre)

# Convertir en numpy array
X = np.array(X)
y = np.array(y)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle SVM
model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)

# Évaluer la précision sur l'ensemble de test
y_pred = model_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Charger le modèle SVM pré-entraîné directement
model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)

@app.route('/svm_service', methods=['POST'])
def svm_service():
    try:
        # Récupérer le fichier audio directement depuis la requête
        wav_music_file = request.files['file']

        # Lire le fichier audio
        wav_music = wav_music_file.read()

        # Faire la prédiction
        prediction = model_svm.predict([np.frombuffer(wav_music, dtype=np.float32)])

        # Renvoyer la prédiction en JSON
        return jsonify({'genre': prediction[0]})
    except Exception as e:
        # En cas d'erreur, renvoyer une réponse d'erreur
        return jsonify({'error': str(e)}), 500



