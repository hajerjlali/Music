# vgg_service.py
from flask import Flask, request, jsonify
import base64
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

# Charger le modèle VGG19 pré-entraîné
model_vgg19 = load_model('../models/vgg19_model.h5')

def preprocess_image(base64_data):
    # Convertir base64 en image
    image_data = base64.b64decode(base64_data)
    image = img_to_array(load_img(io.BytesIO(image_data), target_size=(224, 224)))
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image

@app.route('/vgg_service', methods=['POST'])
def vgg_service():
    data = request.json
    image_base64 = data['image']

    # Prétraitement de l'image
    preprocessed_image = preprocess_image(image_base64)

    # Faire la prédiction
    prediction = model_vgg19.predict(preprocessed_image)

    # Récupérer le genre prédit
    predicted_genre_index = np.argmax(prediction)
    
    # Mappez l'indice prédit vers le genre réel (à adapter en fonction de votre encodage)
    genres = ["classical", "blues", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    predicted_genre = genres[predicted_genre_index]

    return jsonify({'genre': predicted_genre})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

