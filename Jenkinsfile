pipeline {
    agent any

    environment {
        IMAGE_NAME = "dhanasree86/ml-inference-api:latest"
        CONTAINER_NAME = "wine_test_container"
    }

    stages {

        stage('Pull Image') {
            steps {
                sh "docker pull $IMAGE_NAME"
            }
        }

        stage('Run Container') {
            steps {
                sh """
                docker rm -f $CONTAINER_NAME || true
                docker run -d -p 8000:8000 --name $CONTAINER_NAME $IMAGE_NAME
                """
            }
        }

        stage('Wait for Service Readiness') {
            steps {
                script {
                    timeout(time: 90, unit: 'SECONDS') {
                        waitUntil {
                            def status = sh(
                                script: "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health || true",
                                returnStdout: true
                            ).trim()
                            return (status == "200")
                        }
                    }
                }
            }
        }

        stage('Valid Inference Test') {
            steps {
                script {
                    def response = sh(
                        script: """curl -s -X POST http://localhost:8000/predict \
                        -H 'Content-Type: application/json' \
                        -d '{
                              "fixed_acidity": 7.4,
                              "volatile_acidity": 0.7,
                              "citric_acid": 0.0,
                              "residual_sugar": 1.9,
                              "chlorides": 0.076,
                              "pH": 3.51,
                              "sulphates": 0.56,
                              "alcohol": 9.4
                        }'""",
                        returnStdout: true
                    ).trim()

                    echo "Valid Response: ${response}"

                    if (!response.contains("prediction")) {
                        error("Prediction field missing")
                    }
                }
            }
        }

        stage('Invalid Inference Test') {
            steps {
                script {
                    def status = sh(
                        script: """curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:8000/predict \
                        -H 'Content-Type: application/json' \
                        -d '{"fixed_acidity":7.4}'""",
                        returnStdout: true
                    ).trim()

                    echo "Invalid Status Code: ${status}"

                    if (status != "422") {
                        error("Invalid input did not return expected 422 error")
                    }
                }
            }
        }

        stage('Stop Container') {
            steps {
                sh """
                docker stop $CONTAINER_NAME || true
                docker rm $CONTAINER_NAME || true
                """
            }
        }
    }

    post {
        success {
            echo "✅ LAB 7 PASSED – Inference Validation Successful"
        }
        failure {
            echo "❌ LAB 7 FAILED – Validation Error"
        }
        always {
            sh "docker rm -f $CONTAINER_NAME || true"
        }
    }
}
