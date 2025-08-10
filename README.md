# MLOps Pipeline - Iris Classification

A complete MLOps pipeline for the Iris classification dataset demonstrating best practices in model development, tracking, packaging, deployment, and monitoring.

## 🏗️ Architecture Overview

This project implements a full MLOps pipeline with the following components:

- **Data Versioning**: Git + DVC for dataset tracking
- **Experiment Tracking**: MLflow for model experiments and registry
- **API Development**: FastAPI for model serving
- **Containerization**: Docker for consistent deployment
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Monitoring**: Logging and metrics collection

## 📁 Project Structure

```
├── data/                   # Dataset files
├── models/                 # Trained models
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── models/            # Model training
│   ├── api/               # FastAPI application
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── docker/                # Docker configuration
├── .github/               # GitHub Actions workflows
├── mlruns/                # MLflow tracking
├── logs/                  # Application logs
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image
└── docker-compose.yml    # Local deployment
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLOPS_Group_132
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train models and track experiments**
   ```bash
   python src/models/train.py
   ```

4. **Run the API locally**
   ```bash
   python src/api/main.py
   ```

5. **Make predictions**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
   ```

### Docker Deployment

1. **Build and run with Docker**
   ```bash
   docker build -t iris-mlops .
   docker run -p 8000:8000 iris-mlops
   ```

2. **Or use Docker Compose**
   ```bash
   docker-compose up --build
   ```

## 📊 Model Performance

The pipeline trains multiple models and selects the best performing one:

- **Logistic Regression**: Baseline model
- **Random Forest**: Ensemble model with hyperparameter tuning
- **Support Vector Machine**: Advanced classification model

## 🔧 API Endpoints

- `GET /`: Health check
- `POST /predict`: Make predictions
- `GET /metrics`: Model performance metrics
- `GET /logs`: Recent prediction logs

## 📈 Monitoring

- **Logging**: All predictions are logged with timestamps
- **Metrics**: Model performance metrics exposed via `/metrics`
- **Health Checks**: API health monitoring

## 🔄 CI/CD Pipeline

The GitHub Actions workflow:
1. Runs linting and tests
2. Builds Docker image
3. Pushes to Docker Hub
4. Deploys to target environment

## 🛠️ Technologies Used

- **Python**: Core programming language
- **FastAPI**: Modern web framework for APIs
- **MLflow**: Experiment tracking and model registry
- **Docker**: Containerization
- **GitHub Actions**: CI/CD automation
- **Pydantic**: Data validation
- **SQLite**: Logging storage

## 📝 License

MIT License 