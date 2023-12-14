from flask import Flask
from src.svm_service import svm_service
from src.vgg19_service import vgg19_service

app = Flask(__name__)

app.register_blueprint(svm_service, url_prefix='/svm')
app.register_blueprint(vgg19_service, url_prefix='/vgg19')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
