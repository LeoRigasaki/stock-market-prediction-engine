#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 13
FastAPI REST API for Model Serving
"""

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import uvicorn
from contextlib import asynccontextmanager
import logging
from functools import wraps
import hashlib
import secrets

# Internal imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import Config
from src.realtime_prediction import RealTimePredictionEngine
from src.risk_management import RiskManagementFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
prediction_engine = None
risk_framework = None
api_cache = {}
request_counts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global prediction_engine, risk_framework
    
    logger.info("🚀 Starting Stock Market Prediction API...")
    
    # Initialize prediction engine
    try:
        prediction_engine = RealTimePredictionEngine()
        if not prediction_engine.load_production_models():
            logger.error("❌ Failed to load prediction models")
            raise Exception("Model loading failed")
        
        risk_framework = RiskManagementFramework()
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down Stock Market Prediction API...")

# Create FastAPI app
app = FastAPI(
    title="Stock Market Prediction API",
    description="Production-ready ML API for real-time stock predictions and portfolio optimization",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # React, Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.herokuapp.com"]
)

# Pydantic models
class PredictionRequest(BaseModel):
    symbols: List[str] = Field(..., example=["AAPL", "AMZN", "NVDA"], max_items=10)
    include_confidence: bool = Field(default=True, description="Include confidence intervals")
    include_alerts: bool = Field(default=True, description="Include trading alerts")

class SinglePredictionRequest(BaseModel):
    symbol: str = Field(..., example="AAPL", description="Stock symbol")
    horizon: str = Field(default="5d", example="5d", description="Prediction horizon")

class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str] = Field(..., example=["AAPL", "AMZN", "NVDA"], max_items=20)
    optimization_method: str = Field(default="markowitz", example="markowitz")
    target_return: Optional[float] = Field(default=None, example=0.10)
    risk_tolerance: str = Field(default="medium", example="medium")

class AlertSubscription(BaseModel):
    symbols: List[str] = Field(..., max_items=50)
    alert_types: List[str] = Field(default=["high_confidence", "risk_limit"])
    webhook_url: Optional[str] = Field(default=None)

class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    confidence: str
    direction: str
    timestamp: str
    model_version: str

class PortfolioResponse(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    optimization_method: str

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    models_loaded: int
    last_prediction: Optional[str]

# Authentication
API_KEYS = {
    "demo_key_12345": {"name": "Demo User", "tier": "basic", "requests_per_minute": 60},
    "prod_key_67890": {"name": "Production User", "tier": "premium", "requests_per_minute": 300}
}

def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate API key"""
    token = credentials.credentials
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return API_KEYS[token]

# Rate limiting helper
def check_rate_limit(user_info: dict, max_requests: int = 60, window_seconds: int = 60):
    """Check rate limit for user"""
    user_id = user_info['name']
    current_time = time.time()
    
    # Initialize user request tracking
    if user_id not in request_counts:
        request_counts[user_id] = []
    
    # Clean old requests outside window
    request_counts[user_id] = [
        req_time for req_time in request_counts[user_id] 
        if current_time - req_time < window_seconds
    ]
    
    # Check rate limit
    user_limit = user_info.get('requests_per_minute', max_requests)
    if len(request_counts[user_id]) >= user_limit:
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Max {user_limit} requests per minute."
        )
    
    # Record this request
    request_counts[user_id].append(current_time)

# Helper functions
def cache_response(key: str, data: Any, ttl_seconds: int = 300):
    """Cache API response"""
    expiry = time.time() + ttl_seconds
    api_cache[key] = {"data": data, "expiry": expiry}

def get_cached_response(key: str) -> Optional[Any]:
    """Get cached response if valid"""
    if key in api_cache:
        cached = api_cache[key]
        if time.time() < cached["expiry"]:
            return cached["data"]
        else:
            del api_cache[key]
    return None

async def validate_symbols(symbols: List[str]) -> List[str]:
    """Validate and clean stock symbols"""
    valid_symbols = []
    for symbol in symbols:
        # Basic validation
        symbol = symbol.upper().strip()
        if len(symbol) <= 5 and symbol.isalpha():
            valid_symbols.append(symbol)
    return valid_symbols

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Stock Market Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "operational"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global prediction_engine
    
    uptime = time.time() - getattr(app.state, 'start_time', time.time())
    
    return HealthResponse(
        status="healthy" if prediction_engine else "unhealthy",
        version="1.0.0",
        uptime_seconds=uptime,
        models_loaded=len(prediction_engine.models) if prediction_engine else 0,
        last_prediction=getattr(app.state, 'last_prediction', None)
    )

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_stocks(
    request: PredictionRequest,
    current_user: dict = Depends(get_api_key)
):
    """Generate predictions for multiple stocks"""
    global prediction_engine
    
    # Check rate limit
    check_rate_limit(current_user, max_requests=30)
    
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="Prediction service unavailable")
    
    try:
        # Validate symbols
        valid_symbols = await validate_symbols(request.symbols)
        if not valid_symbols:
            raise HTTPException(status_code=400, detail="No valid symbols provided")
        
        # Check cache
        cache_key = f"predict_{hash(tuple(sorted(valid_symbols)))}"
        cached_result = get_cached_response(cache_key)
        if cached_result:
            return cached_result
        
        # Run prediction
        logger.info(f"🔮 Generating predictions for {len(valid_symbols)} symbols")
        
        # Temporarily override target stocks for API call
        original_stocks = prediction_engine.get_target_stocks
        prediction_engine.get_target_stocks = lambda: valid_symbols
        
        # Run prediction cycle
        results = await prediction_engine.run_realtime_cycle()
        
        # Restore original method
        prediction_engine.get_target_stocks = original_stocks
        
        # Format response
        predictions = []
        for symbol, pred_data in results.get('predictions', {}).items():
            primary = pred_data.get('primary', {})
            prediction_value = primary.get('prediction', 0)
            
            predictions.append(PredictionResponse(
                symbol=symbol,
                prediction=round(prediction_value, 6),
                confidence=primary.get('confidence', 'medium'),
                direction="BUY" if prediction_value > 0.001 else "SELL" if prediction_value < -0.001 else "HOLD",
                timestamp=pred_data.get('timestamp', datetime.now().isoformat()),
                model_version="ensemble_v1.0"
            ))
        
        # Cache result
        cache_response(cache_key, predictions, ttl_seconds=300)  # 5 minutes
        
        # Update app state
        app.state.last_prediction = datetime.now().isoformat()
        
        logger.info(f"✅ Generated {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single_stock(
    request: SinglePredictionRequest,
    current_user: dict = Depends(get_api_key)
):
    """Generate prediction for a single stock"""
    # Check rate limit
    check_rate_limit(current_user, max_requests=60)
    prediction_request = PredictionRequest(
        symbols=[request.symbol],
        include_confidence=True,
        include_alerts=False
    )
    
    predictions = await predict_stocks(prediction_request, current_user)
    
    if not predictions:
        raise HTTPException(status_code=404, detail=f"No prediction available for {request.symbol}")
    
    return predictions[0]

@app.post("/portfolio/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(
    request: PortfolioOptimizationRequest,
    current_user: dict = Depends(get_api_key)
):
    """Optimize portfolio allocation"""
    global risk_framework
    
    # Check rate limit
    check_rate_limit(current_user, max_requests=10)
    
    if not risk_framework:
        raise HTTPException(status_code=503, detail="Portfolio optimization service unavailable")
    
    try:
        # Validate symbols
        valid_symbols = await validate_symbols(request.symbols)
        if len(valid_symbols) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid symbols for optimization")
        
        # Check cache
        cache_key = f"portfolio_{hash(tuple(sorted(valid_symbols)))}_{request.optimization_method}"
        cached_result = get_cached_response(cache_key)
        if cached_result:
            return cached_result
        
        logger.info(f"📊 Optimizing portfolio for {len(valid_symbols)} symbols")
        
        # Load feature data (mock for API - in production would fetch real-time)
        config = Config()
        features_path = config.FEATURES_DATA_PATH / "selected_features.csv"
        
        if not features_path.exists():
            raise HTTPException(status_code=503, detail="Feature data unavailable")
        
        df = pd.read_csv(features_path)
        
        # Filter for requested symbols
        df_filtered = df[df['Ticker'].isin(valid_symbols)].copy()
        
        if df_filtered.empty:
            raise HTTPException(status_code=404, detail="No data available for requested symbols")
        
        # Generate predictions for portfolio optimization
        X, y, feature_cols = risk_framework.prepare_portfolio_data(df_filtered)
        predictions_df = risk_framework.generate_predictions(risk_framework.load_best_models(), X, df_filtered)
        
        # Run optimization
        if request.optimization_method == "markowitz":
            result = risk_framework.portfolio_optimization_markowitz(
                predictions_df, target_return=request.target_return
            )
        else:  # risk_parity
            result = risk_framework.risk_parity_portfolio(predictions_df)
        
        if not result or not result.get('success', True):
            raise HTTPException(status_code=400, detail="Portfolio optimization failed")
        
        # Format response
        portfolio_response = PortfolioResponse(
            weights=dict(zip(result['stocks'], result['weights'])) if 'stocks' in result else {},
            expected_return=result.get('expected_return', result.get('portfolio_return', 0)),
            volatility=result.get('volatility', result.get('portfolio_volatility', 0)),
            sharpe_ratio=result.get('sharpe_ratio', 0),
            optimization_method=request.optimization_method
        )
        
        # Cache result
        cache_response(cache_key, portfolio_response, ttl_seconds=600)  # 10 minutes
        
        logger.info(f"✅ Portfolio optimization completed")
        return portfolio_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/models/performance", response_model=Dict[str, Any])
async def get_model_performance(current_user: dict = Depends(get_api_key)):
    """Get model performance metrics"""
    # Check rate limit
    check_rate_limit(current_user, max_requests=20)
    try:
        config = Config()
        
        # Load performance data from Day 11
        risk_summary_path = config.PROCESSED_DATA_PATH / "day11_risk_summary.csv"
        
        if not risk_summary_path.exists():
            raise HTTPException(status_code=404, detail="Performance data not available")
        
        risk_df = pd.read_csv(risk_summary_path)
        
        # Convert to response format
        performance_data = {
            "models": risk_df.to_dict('records'),
            "best_model": risk_df.loc[risk_df['Sharpe_Ratio'].idxmax()].to_dict(),
            "summary": {
                "total_models": len(risk_df),
                "avg_sharpe": risk_df['Sharpe_Ratio'].mean(),
                "best_sharpe": risk_df['Sharpe_Ratio'].max(),
                "avg_annual_return": risk_df['Annual_Return'].mean() * 100,
                "avg_max_drawdown": risk_df['Max_Drawdown'].mean()
            },
            "last_updated": datetime.now().isoformat()
        }
        
        return performance_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get performance data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance data")

@app.get("/alerts/active", response_model=List[Dict[str, Any]])
async def get_active_alerts(current_user: dict = Depends(get_api_key)):
    """Get active trading alerts"""
    global prediction_engine
    
    # Check rate limit
    check_rate_limit(current_user, max_requests=30)
    
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="Alert service unavailable")
    
    try:
        # Run quick prediction cycle to get latest alerts
        results = await prediction_engine.run_realtime_cycle()
        alerts = results.get('alerts', [])
        
        # Format alerts for API response
        formatted_alerts = []
        for alert in alerts:
            formatted_alerts.append({
                "id": hashlib.md5(f"{alert.get('symbol', '')}{alert.get('timestamp', '')}".encode()).hexdigest()[:8],
                "symbol": alert.get('symbol'),
                "type": alert.get('type'),
                "message": alert.get('message'),
                "severity": "high" if alert.get('type') == 'risk_limit_exceeded' else "medium",
                "timestamp": alert.get('timestamp'),
                "data": alert
            })
        
        return formatted_alerts
        
    except Exception as e:
        logger.error(f"❌ Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@app.get("/market/status", response_model=Dict[str, Any])
async def get_market_status():
    """Get overall market status (no auth required)"""
    try:
        now = datetime.now()
        
        # Simple market hours check (NYSE: 9:30 AM - 4:00 PM EST)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_market_hours = market_open <= now <= market_close and now.weekday() < 5
        
        return {
            "is_open": is_market_hours,
            "current_time": now.isoformat(),
            "next_open": market_open.isoformat() if not is_market_hours else None,
            "next_close": market_close.isoformat() if is_market_hours else None,
            "timezone": "UTC",
            "trading_session": "regular" if is_market_hours else "closed"
        }
        
    except Exception as e:
        return {"error": str(e), "is_open": False}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Record startup time"""
    app.state.start_time = time.time()
    logger.info("🚀 Stock Market Prediction API started successfully")

# Main function for running the server
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )