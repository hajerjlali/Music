
from flask import Flask, request, jsonify,render_template
import os
import pickle
import joblib

import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

import io


app = Flask(__name__,static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Adjust as needed
app.config['UPLOAD_FOLDER'] = '../upload'
file_name = 'Vgg19Keras.pkl'

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# Create the full file path
file_path = os.path.join(current_directory, file_name)
# Load the content of the file using joblib
model = joblib.load(file_path)

# Create the full file path
file_path = os.path.join(current_directory, 'standard_scaler.pkl')
# Load the content of the file using joblib
scaler = joblib.load(file_path)

@app.route('/vgg', methods=['POST'])
def svm():
     if request.method == 'POST':
          request_data = request.get_json()
          print(request.get_json())
          data_samples = np.array(request_data)
          data_samples_normalized = scaler.transform(data_samples)
          prediction=model.predict(data_samples_normalized)
          return jsonify("Your music file type is "+prediction[0])
               
if __name__ == '__main__':
          app.run(debug=True)
