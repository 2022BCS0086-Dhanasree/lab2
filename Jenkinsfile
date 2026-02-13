pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "dhanasree86/ml-model"
    }

    stages {

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}")
                }
            }
        }

        stage('Train and Evaluate Model') {
            steps {
                sh 'docker run --rm dhanasree86/ml-model python train.py'
            }
        }

        stage('Print Metrics with Details') {
            steps {
                sh '''
                echo "Name: Dhanasree Gidijala"
                echo "Roll No: 2022BCS0086"
                cat metrics.txt
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('', 'dockerhub-credentials') {
                        docker.image("${DOCKER_IMAGE}").push("latest")
                    }
                }
            }
        }

    }
}