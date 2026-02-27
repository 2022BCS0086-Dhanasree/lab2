pipeline {
    agent any

    environment {
        IMAGE = "dhanasree86/ml-inference-api:latest"
        CONTAINER = "wine_test_container"
        PORT = "8000"
    }

    stages {

        stage('Pull Image') {
            steps {
                sh "docker pull $IMAGE"
            }
        }

        stage('Run Container') {
            steps {
                sh """
                docker run -d -p $PORT:8000 --name $CONTAINER $IMAGE
                """
            }
        }

        stage('Wait for Service Readiness') {
            steps {
                script {
                    timeout(time: 60, unit: 'SECONDS') {
                        waitUntil {
                            def status = sh(
                                script: "curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/docs || true",
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
                        script: """
                        curl -s -X POST http://localhost:$PORT/predict \
                        -H "Content-Type: application/json" \
                        -d @tests/valid.json
                        """,
                        returnStdout: true
                    ).trim()

                    echo "Valid Response: ${response}"

                    if (!response.contains("wine_quality")) {
                        error("wine_quality field missing!")
                    }

                    if (!response.contains("2022BCS0086")) {
                        error("Roll number missing in response!")
                    }
                }
            }
        }

        stage('Invalid Input Test') {
            steps {
                script {
                    def code = sh(
                        script: """
                        curl -s -o /dev/null -w "%{http_code}" \
                        -X POST http://localhost:$PORT/predict \
                        -H "Content-Type: application/json" \
                        -d @tests/invalid.json
                        """,
                        returnStdout: true
                    ).trim()

                    echo "Invalid Request Status Code: ${code}"

                    if (code == "200") {
                        error("Invalid request should not succeed")
                    }
                }
            }
        }

        stage('Stop Container') {
            steps {
                sh """
                docker stop $CONTAINER || true
                docker rm $CONTAINER || true
                """
            }
        }
    }

    post {
        success {
            echo "✅ LAB 7 PASSED – Inference Validation Successful"
        }
        failure {
            echo "❌ LAB 7 FAILED – Validation Error Detected"
        }
    }
}
