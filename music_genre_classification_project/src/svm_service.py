# svm_service.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import pickle
import os
import librosa

app = Flask(__name__)
CORS(app)

# Récupérer le chemin du répertoire courant
current_directory = os.path.dirname(os.path.realpath(__file__))

# Charger le modèle SVM pré-entraîné
model_svm_path = '/app/models/svm_model.pkl'
with open(model_svm_path,'rb') as model_file:
  model_svm = pickle.load(model_file)
print('avant')

@app.route('/svm_service', methods=['POST'])
def svm_service():
    global resultat_prediction

    # Assurez-vous que la clé 'wav_music' est utilisée pour envoyer le fichier audio
    if 'wav_music' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['wav_music']

    # Assurez-vous que le fichier est un fichier WAV
    if audio_file and audio_file.filename.endswith('.wav'):
        # Lire le contenu du fichier audio
        wav_music, _ = librosa.load(audio_file, sr=None, mono=True, dtype=np.float32)
        # Appliquer le même prétraitement que lors de l'entraînement
        mfccs = np.mean(librosa.feature.mfcc(y=wav_music, sr=_, n_mfcc=13), axis=1)


        print('Forme du tableau wav_music:', wav_music.shape)

        # Faire la prédiction
        if np.isnan(mfccs).any():
             error_message = {
                   'error': 'Audio file contains NaN values',
                   'array_shape': mfccs.shape,
                   'content_length': len(audio_content)
             }
             return jsonify(error_message), 400

        prediction = model_svm.predict([mfccs])

        resultat_prediction = {'genre': prediction[0]}
        return jsonify(resultat_prediction)
    else:
        return jsonify({'error': 'Invalid audio file format'}), 400

@app.route('/get_result', methods=['GET'])
def get_result():
    global resultat_prediction

    if resultat_prediction:
        return jsonify(resultat_prediction)
    else:
        return jsonify({'error': 'No result available'}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


