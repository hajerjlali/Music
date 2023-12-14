#start.sh
app="docker.testvgg33"
# Construire l'image Docker
docker build -t ${app} .

# Exécuter le conteneur Docker
docker run -p 5001:5001 -d  ${app}

# Attendre quelques secondes pour que le serveur démarre
sleep 10


