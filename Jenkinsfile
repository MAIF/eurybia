pipeline {
    agent {
        docker {
            image 'nexus-fabfonc.maif.local:5000/fr-maif-datalab/python-builder:3.6.5'
        }
    }
    stages {
        stage('Installing dependencies') {
            steps {
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh 'python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org --user --no-cache'
                    sh 'python -m pip install .[testing] --index-url http://nexus-fabfonc.maif.local/repository/pip-group/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host nexus-fabfonc.maif.local --user --no-cache'
                }
            }
        }
        stage('Test python package') {
            steps {
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh 'python -m pytest'
                }
            }
        }
        stage('Build python package') {
            steps {
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh 'python setup.py bdist_wheel'
                }
            }
        }
        stage('Deploy python package') {
            steps {
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh 'python -m twine upload -u deploy-pip -p deploy-pip --repository-url http://nexus-fabfonc.maif.local/repository/pip-hosted/ dist/*'
                }
            }
        }
    }
}
