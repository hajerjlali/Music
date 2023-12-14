pipeline {
    agent any

    stages {
        stage('Build SVM Service') {
            steps {
                script {
                    // Étape pour construire le service SVM
                    dir('music_genre_classification_project') {
                        sh 'docker build -t svm_service .'
                    }
                }
            }
        }

        stage('Build Front Service') {
            steps {
                script {
                    // Étape pour construire le service Front
                    dir('TestApp') {
                        sh 'docker build -t front_service .'
                    }
                }
            }
        }

        stage('Deploy to Docker Compose') {
            steps {
                script {
                    // Étape pour lancer Docker Compose
                    sh 'docker-compose up -d'
                }
            }
        }
        
        // Ajoutez d'autres étapes au besoin (tests, déploiement, etc.).
    }

    post {
        always {
            // Étape pour arrêter Docker Compose après l'exécution des étapes
            sh 'docker-compose down'
        }
    }
}
