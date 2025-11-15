pipeline {
    agent any

    environment {
        IMAGE_NAME = "fastapi-matrix-app"
        CONTAINER_NAME = "fastapi-matrix-container"
        PORT = "8613"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/abdoukhemir/cuda-soa-lab.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh "docker build -t ${IMAGE_NAME}:latest ."
                }
            }
        }

        stage('Stop Old Container') {
            steps {
                script {
                    sh """
                        if [ \$(docker ps -q -f name=${CONTAINER_NAME}) ]; then
                            docker stop ${CONTAINER_NAME}
                            docker rm ${CONTAINER_NAME}
                        fi
                    """
                }
            }
        }

        stage('Run New Container') {
            steps {
                script {
                    sh "docker run -d --name ${CONTAINER_NAME} -p ${PORT}:${PORT} ${IMAGE_NAME}:latest"
                }
            }
        }

        stage('Health Check') {
            steps {
                script {
                    sh "curl -s http://localhost:${PORT}/health"
                }
            }
        }
    }

    post {
        success {
            echo "Deployment succeeded! App is running on port ${PORT}"
        }
        failure {
            echo "Deployment failed!"
        }
    }
}
