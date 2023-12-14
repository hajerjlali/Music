import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import to_categorical
import os

# Charger les données d'images
data_dir = '/app/data/GTZAN/images_original'
genres = os.listdir(data_dir)

X = []
y = []

# Taille d'image souhaitée pour VGG19
target_size = (224, 224)

for genre in genres:
    genre_dir = os.path.join(data_dir, genre)
    for filename in os.listdir(genre_dir):
        file_path = os.path.join(genre_dir, filename)
        # Charger l'image et la prétraiter pour VGG19
        img = image.load_img(file_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))
        X.append(img_array)
        y.append(genre)

X = np.array(X)
y = np.array(y)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Charger le modèle VGG19
vgg_model = VGG19(weights='imagenet', include_top=False)

# Extraire les caractéristiques des images
X_train_features = vgg_model.predict(X_train)
X_test_features = vgg_model.predict(X_test)

# Aplatir les données pour SVM
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# Convertir les labels en one-hot encoding
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# Entraîner le modèle SVM
model_svm = SVC(kernel='linear')
model_svm.fit(X_train_features, y_train_onehot)

# Évaluer la précision sur l'ensemble de test
y_pred_onehot = model_svm.predict(X_test_features)
y_pred = np.argmax(y_pred_onehot, axis=1)
accuracy = accuracy_score(np.argmax(y_test_onehot, axis=1), y_pred)
print(f'Accuracy: {accuracy}')

# Enregistrer le modèle SVM
model_svm_path = '/app/models/vgg_model.pkl'
with open(model_svm_path, 'wb') as model_file:
    pickle.dump(model_svm, model_file)

