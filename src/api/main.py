"""
FastAPI application for Iris classification model serving
"""
import os
import sys
import logging
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.loader import IrisDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
Path(os.path.join(os.path.dirname(__file__), '..', '..', 'logs')).mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="MLOps pipeline for Iris flower classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics - use unique names to avoid conflicts during reload
PREDICTION_COUNTER = Counter('iris_predictions_total', 'Total number of predictions')
PREDICTION_DURATION = Histogram('iris_prediction_duration_seconds', 'Time spent processing prediction')
ERROR_COUNTER = Counter('iris_prediction_errors_total', 'Total number of prediction errors')

# Pydantic models for request/response validation
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")
    
    @validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    def validate_measurements(cls, v):
        if v <= 0:
            raise ValueError('Measurement must be positive')
        if v > 10:
            raise ValueError('Measurement seems too large for Iris flowers')
        return v

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    model_info: Dict[str, Any]

class LogEntry(BaseModel):
    timestamp: str
    features: Dict[str, float]
    prediction: str
    confidence: float
    processing_time: float

# Global variables
model = None
model_info = {}
scaler = None
target_names = ['setosa', 'versicolor', 'virginica']
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

class PredictionLogger:
    """SQLite-based prediction logger"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'predictions.db')
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sepal_length REAL NOT NULL,
                sepal_width REAL NOT NULL,
                petal_length REAL NOT NULL,
                petal_width REAL NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                processing_time REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def log_prediction(self, features: Dict[str, float], prediction: str, 
                      confidence: float, processing_time: float):
        """Log a prediction to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, sepal_length, sepal_width, petal_length, petal_width, 
             prediction, confidence, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            features['sepal_length'],
            features['sepal_width'],
            features['petal_length'],
            features['petal_width'],
            prediction,
            confidence,
            processing_time
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_logs(self, limit: int = 10) -> List[LogEntry]:
        """Get recent prediction logs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, sepal_length, sepal_width, petal_length, petal_width,
                   prediction, confidence, processing_time
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            logs.append(LogEntry(
                timestamp=row[0],
                features={
                    'sepal_length': row[1],
                    'sepal_width': row[2],
                    'petal_length': row[3],
                    'petal_width': row[4]
                },
                prediction=row[5],
                confidence=row[6],
                processing_time=row[7]
            ))
        
        return logs

# Initialize logger
prediction_logger = PredictionLogger()

def load_model():
    """Load the trained model"""
    global model, model_info, scaler
    
    try:
        # Load model
        model_path = Path(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best_model.pkl'))
        if not model_path.exists():
            logger.error("Model file not found. Please train the model first.")
            return False
        
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        # Load model info
        info_path = Path(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'model_info.json'))
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            logger.info(f"Model info loaded: {model_info}")
        
        # Initialize scaler (same as used in training)
        data_loader = IrisDataLoader()
        _, _, _, _, _, _ = data_loader.load_data()
        scaler = data_loader.scaler
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Iris Classification API...")
    
    if not load_model():
        logger.error("Failed to load model. API may not function correctly.")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        model_info=model_info
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, features: IrisFeatures):
    """Make a prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Convert features to numpy array
        feature_values = np.array([
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]).reshape(1, -1)
        
        # Scale features
        if scaler is not None:
            feature_values = scaler.transform(feature_values)
        
        # Make prediction
        prediction_proba = model.predict_proba(feature_values)[0]
        prediction_idx = model.predict(feature_values)[0]
        prediction_class = target_names[prediction_idx]
        confidence = float(max(prediction_proba))
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log prediction
        prediction_logger.log_prediction(
            features=features.dict(),
            prediction=prediction_class,
            confidence=confidence,
            processing_time=processing_time
        )
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_DURATION.observe(processing_time)
        
        # Prepare response
        probabilities = dict(zip(target_names, prediction_proba.tolist()))
        
        logger.info(f"Prediction made: {prediction_class} (confidence: {confidence:.3f})")
        
        return PredictionResponse(
            prediction=prediction_class,
            confidence=confidence,
            probabilities=probabilities,
            model_info=model_info
        )
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/logs", response_model=List[LogEntry])
async def get_logs(limit: int = 10):
    """Get recent prediction logs"""
    if limit > 100:
        limit = 100  # Cap at 100 entries
    
    logs = prediction_logger.get_recent_logs(limit)
    return logs

@app.get("/sample")
async def get_sample_data():
    """Get sample data for testing"""
    data_loader = IrisDataLoader()
    sample = data_loader.get_sample_data()
    return {
        "sample_data": sample,
        "description": "Sample Iris flower measurements for testing"
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {processing_time:.3f}s"
    )
    
    return response

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 