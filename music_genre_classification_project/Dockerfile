# Utilisez l'image de base Python
FROM python:3.8

# Mise à jour des paquets
RUN apt-get update && apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# Mettre à jour pip
RUN pip install --upgrade pip

# Définit le répertoire de travail
WORKDIR /app

RUN pip install flask-cors

# Copie les fichiers nécessaires dans le conteneur
COPY . /app

# Installe les dépendances
RUN pip install --timeout=1000  --no-cache-dir -r requirements.txt

# Expose le port nécessaire (facultatif, dépend de votre application)
EXPOSE 5000

# Définir l'exécution du script de démarrage
CMD ["python","src/svm_service.py"]
