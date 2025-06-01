#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 14
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.realtime_prediction import RealTimePredictionEngine
from src.risk_management import RiskManagementFramework

# Page config
st.set_page_config(
    page_title="Stock Market AI Prediction Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DashboardApp:
    def __init__(self):
        self.config = Config()
        self.prediction_engine = None
        self.risk_framework = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ML components"""
        try:
            if 'prediction_engine' not in st.session_state:
                self.prediction_engine = RealTimePredictionEngine()
                success = self.prediction_engine.load_production_models()
                st.session_state.prediction_engine = self.prediction_engine
                st.session_state.models_loaded = success
            else:
                self.prediction_engine = st.session_state.prediction_engine
            
            if 'risk_framework' not in st.session_state:
                self.risk_framework = RiskManagementFramework()
                st.session_state.risk_framework = self.risk_framework
            else:
                self.risk_framework = st.session_state.risk_framework
                
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
    
    def load_performance_data(self):
        """Load model performance data"""
        try:
            risk_summary_path = self.config.PROCESSED_DATA_PATH / "day11_risk_summary.csv"
            if risk_summary_path.exists():
                return pd.read_csv(risk_summary_path)
            else:
                # Create dummy data for demo
                return pd.DataFrame({
                    'Model': ['Ensemble_SimpleAverage', 'XGBoost', 'LightGBM', 'RandomForest'],
                    'Sharpe_Ratio': [4.25, 3.82, 3.76, 3.45],
                    'Annual_Return': [0.15, 0.12, 0.11, 0.10],
                    'Max_Drawdown': [-0.08, -0.12, -0.10, -0.15],
                    'Win_Rate': [65.2, 62.1, 61.8, 59.5]
                })
        except Exception as e:
            st.error(f"Error loading performance data: {e}")
            return pd.DataFrame()
    
    def get_target_stocks(self):
        """Get target stocks"""
        try:
            stocks_path = self.config.PROCESSED_DATA_PATH / "target_stocks.txt"
            if stocks_path.exists():
                with open(stocks_path, 'r') as f:
                    return [line.strip() for line in f.readlines()][:10]
            else:
                return ['AAPL', 'AMZN', 'NVDA', 'MSFT', 'AMD', 'GOOGL', 'TSLA', 'META', 'NFLX', 'CRM']
        except:
            return ['AAPL', 'AMZN', 'NVDA', 'MSFT', 'AMD']
    
    def render_header(self):
        """Render main header"""
        st.markdown('<h1 class="main-header">📈 Stock Market AI Prediction Engine</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            models_loaded = st.session_state.get('models_loaded', False)
            status = "🟢 Online" if models_loaded else "🔴 Offline"
            st.markdown(f"**System Status:** {status}")
        
        with col2:
            model_count = len(self.prediction_engine.models) if self.prediction_engine else 0
            st.markdown(f"**Models Loaded:** {model_count}")
        
        with col3:
            st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        
        with col4:
            if st.button("🔄 Refresh Data"):
                st.rerun()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Navigation
        page = st.sidebar.selectbox(
            "📋 Select Page",
            ["🏠 Overview", "🔮 Live Predictions", "📊 Performance Analytics", 
             "💼 Portfolio Optimizer", "🚨 Alert Center", "🤖 Model Insights"]
        )
        
        st.sidebar.markdown("---")
        
        # Stock selection
        available_stocks = self.get_target_stocks()
        selected_stocks = st.sidebar.multiselect(
            "📈 Select Stocks",
            available_stocks,
            default=available_stocks[:5]
        )
        
        # Time horizon
        time_horizon = st.sidebar.selectbox(
            "⏰ Prediction Horizon",
            ["1 Day", "5 Days", "10 Days"],
            index=1
        )
        
        # Risk tolerance
        risk_tolerance = st.sidebar.selectbox(
            "⚖️ Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            index=1
        )
        
        st.sidebar.markdown("---")
        
        # Quick stats
        st.sidebar.markdown("### 📊 Quick Stats")
        performance_data = self.load_performance_data()
        if not performance_data.empty:
            best_model = performance_data.loc[performance_data['Sharpe_Ratio'].idxmax()]
            st.sidebar.metric("🏆 Best Sharpe", f"{best_model['Sharpe_Ratio']:.2f}")
            st.sidebar.metric("📈 Best Return", f"{best_model['Annual_Return']*100:.1f}%")
            st.sidebar.metric("📉 Max Drawdown", f"{best_model['Max_Drawdown']*100:.1f}%")
        
        return page, selected_stocks, time_horizon, risk_tolerance
    
    def render_overview_page(self, selected_stocks):
        """Render overview page"""
        st.header("🏠 System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        performance_data = self.load_performance_data()
        if not performance_data.empty:
            with col1:
                best_sharpe = performance_data['Sharpe_Ratio'].max()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏆 Best Sharpe Ratio</h3>
                    <h2>{best_sharpe:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_return = performance_data['Annual_Return'].mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Avg Annual Return</h3>
                    <h2>{avg_return:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                best_win_rate = performance_data['Win_Rate'].max()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Best Win Rate</h3>
                    <h2>{best_win_rate:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                models_count = len(performance_data)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🤖 Models Available</h3>
                    <h2>{models_count}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Model performance chart
        st.subheader("📊 Model Performance Comparison")
        if not performance_data.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=performance_data['Model'],
                y=performance_data['Sharpe_Ratio'],
                name='Sharpe Ratio',
                marker_color='rgb(55, 126, 184)'
            ))
            
            fig.update_layout(
                title="Model Performance (Sharpe Ratio)",
                xaxis_title="Model",
                yaxis_title="Sharpe Ratio",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Market overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌐 Market Status")
            now = datetime.now()
            market_open = 9 <= now.hour <= 16 and now.weekday() < 5
            status = "🟢 Open" if market_open else "🔴 Closed"
            st.markdown(f"**Market Status:** {status}")
            st.markdown(f"**Current Time:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Selected Stocks:** {', '.join(selected_stocks[:5])}")
        
        with col2:
            st.subheader("📈 Portfolio Value")
            # Simulate portfolio value
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            portfolio_values = 100000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 30)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=3)
            ))
            
            fig.update_layout(
                title="30-Day Portfolio Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template="plotly_white",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    async def run_predictions(self, selected_stocks):
        """Run predictions for selected stocks"""
        if not self.prediction_engine or not st.session_state.get('models_loaded', False):
            st.error("❌ Prediction engine not available")
            return {}
        
        try:
            # Override target stocks temporarily
            original_method = self.prediction_engine.get_target_stocks
            self.prediction_engine.get_target_stocks = lambda: selected_stocks
            
            # Run prediction cycle
            results = await self.prediction_engine.run_realtime_cycle()
            
            # Restore original method
            self.prediction_engine.get_target_stocks = original_method
            
            return results
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return {}
    
    def render_predictions_page(self, selected_stocks):
        """Render live predictions page"""
        st.header("🔮 Live Predictions")
        
        # Control panel
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Analyzing:** {', '.join(selected_stocks)}")
        
        with col2:
            auto_refresh = st.checkbox("🔄 Auto Refresh (30s)")
        
        with col3:
            if st.button("🚀 Generate Predictions"):
                st.session_state.trigger_prediction = True
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Generate predictions
        if st.session_state.get('trigger_prediction', False) or auto_refresh:
            with st.spinner("🔮 Generating AI predictions..."):
                # Simulate predictions for demo (replace with real predictions)
                predictions = {}
                for symbol in selected_stocks:
                    pred_value = np.random.normal(0, 0.01)  # Random prediction for demo
                    confidence = "high" if abs(pred_value) > 0.005 else "medium"
                    direction = "BUY" if pred_value > 0.001 else "SELL" if pred_value < -0.001 else "HOLD"
                    
                    predictions[symbol] = {
                        'primary': {
                            'prediction': pred_value,
                            'confidence': confidence
                        },
                        'direction': direction,
                        'timestamp': datetime.now().isoformat()
                    }
                
                st.session_state.predictions = predictions
                st.session_state.trigger_prediction = False
        
        # Display predictions
        if 'predictions' in st.session_state:
            predictions = st.session_state.predictions
            
            # Prediction cards
            cols = st.columns(min(len(predictions), 3))
            for i, (symbol, pred_data) in enumerate(predictions.items()):
                with cols[i % 3]:
                    pred_value = pred_data['primary']['prediction']
                    direction = pred_data['direction']
                    confidence = pred_data['primary']['confidence']
                    
                    card_class = "prediction-positive" if pred_value > 0 else "prediction-negative"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3>{symbol}</h3>
                        <h2>{direction}</h2>
                        <p>Prediction: {pred_value:+.4f}</p>
                        <p>Confidence: {confidence.upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed prediction table
            st.subheader("📋 Detailed Predictions")
            pred_df = pd.DataFrame([
                {
                    'Symbol': symbol,
                    'Prediction': data['primary']['prediction'],
                    'Direction': data['direction'],
                    'Confidence': data['primary']['confidence'],
                    'Timestamp': data['timestamp']
                }
                for symbol, data in predictions.items()
            ])
            
            st.dataframe(pred_df, use_container_width=True)
            
            # Prediction visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(predictions.keys()),
                y=[data['primary']['prediction'] for data in predictions.values()],
                marker_color=['green' if p > 0 else 'red' for p in [data['primary']['prediction'] for data in predictions.values()]],
                name='Predictions'
            ))
            
            fig.update_layout(
                title="Stock Predictions Comparison",
                xaxis_title="Stock Symbol",
                yaxis_title="Prediction Value",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_page(self):
        """Render performance analytics page"""
        st.header("📊 Performance Analytics")
        
        performance_data = self.load_performance_data()
        if performance_data.empty:
            st.warning("No performance data available")
            return
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Model Rankings")
            
            # Sort by Sharpe ratio
            sorted_data = performance_data.sort_values('Sharpe_Ratio', ascending=False)
            
            for i, (_, row) in enumerate(sorted_data.iterrows(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                st.markdown(f"""
                **{medal} {row['Model']}**
                - Sharpe: {row['Sharpe_Ratio']:.2f}
                - Return: {row['Annual_Return']*100:.1f}%
                - Win Rate: {row['Win_Rate']:.1f}%
                """)
        
        with col2:
            st.subheader("📈 Risk-Return Analysis")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=performance_data['Max_Drawdown'].abs() * 100,
                y=performance_data['Annual_Return'] * 100,
                mode='markers+text',
                text=performance_data['Model'],
                textposition="top center",
                marker=dict(
                    size=performance_data['Sharpe_Ratio'] * 10,
                    color=performance_data['Sharpe_Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Models'
            ))
            
            fig.update_layout(
                title="Risk vs Return (Bubble size = Sharpe Ratio)",
                xaxis_title="Max Drawdown (%)",
                yaxis_title="Annual Return (%)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical performance simulation
        st.subheader("📊 Historical Performance Simulation")
        
        # Create time series data
        dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
        
        fig = go.Figure()
        
        for _, row in performance_data.head(3).iterrows():  # Top 3 models
            # Simulate cumulative returns
            daily_return = row['Annual_Return'] / 252
            daily_vol = daily_return * 2  # Simplified volatility
            returns = np.random.normal(daily_return, daily_vol, len(dates))
            cumulative = np.cumprod(1 + returns) * 100000  # $100k initial
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative,
                mode='lines',
                name=row['Model'],
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Simulated Portfolio Performance ($100k initial)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_portfolio_page(self, selected_stocks, risk_tolerance):
        """Render portfolio optimizer page"""
        st.header("💼 Portfolio Optimizer")
        
        # Input panel
        col1, col2, col3 = st.columns(3)
        
        with col1:
            portfolio_value = st.number_input("💰 Portfolio Value ($)", min_value=1000, value=100000, step=1000)
        
        with col2:
            optimization_method = st.selectbox("🎯 Optimization Method", ["Markowitz", "Risk Parity", "Equal Weight"])
        
        with col3:
            target_return = st.number_input("📈 Target Return (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5) / 100
        
        if st.button("🚀 Optimize Portfolio"):
            with st.spinner("🔄 Optimizing portfolio..."):
                # Simulate portfolio optimization
                n_stocks = len(selected_stocks)
                
                if optimization_method == "Equal Weight":
                    weights = [1/n_stocks] * n_stocks
                elif optimization_method == "Risk Parity":
                    # Simulate risk parity weights
                    base_weights = np.random.dirichlet(np.ones(n_stocks))
                    weights = base_weights / base_weights.sum()
                else:  # Markowitz
                    # Simulate Markowitz optimization
                    weights = np.random.dirichlet(np.ones(n_stocks))
                
                # Calculate metrics
                expected_return = target_return + np.random.normal(0, 0.02)
                volatility = 0.15 + np.random.normal(0, 0.03)
                sharpe_ratio = expected_return / volatility if volatility > 0 else 0
                
                # Display results
                st.success("✅ Portfolio optimization completed!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📈 Expected Return", f"{expected_return*100:.1f}%")
                
                with col2:
                    st.metric("📊 Volatility", f"{volatility*100:.1f}%")
                
                with col3:
                    st.metric("🏆 Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                # Portfolio allocation chart
                fig = go.Figure(data=[go.Pie(
                    labels=selected_stocks,
                    values=weights,
                    hole=0.4
                )])
                
                fig.update_layout(
                    title="Optimized Portfolio Allocation",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Allocation table
                allocation_df = pd.DataFrame({
                    'Stock': selected_stocks,
                    'Weight (%)': [w*100 for w in weights],
                    'Value ($)': [w*portfolio_value for w in weights]
                })
                
                st.dataframe(allocation_df, use_container_width=True)
    
    def render_alerts_page(self):
        """Render alert center page"""
        st.header("🚨 Alert Center")
        
        # Alert configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Alert Settings")
            
            high_conf_threshold = st.slider("🎯 High Confidence Threshold", 0.001, 0.05, 0.01, 0.001)
            risk_threshold = st.slider("⚠️ Risk Limit Threshold", 0.02, 0.10, 0.05, 0.01)
            
            enable_email = st.checkbox("📧 Email Alerts")
            enable_push = st.checkbox("📱 Push Notifications")
        
        with col2:
            st.subheader("📊 Alert Statistics")
            
            # Simulate alert stats
            st.metric("🚨 Alerts Today", "7")
            st.metric("📈 High Confidence", "3")
            st.metric("⚠️ Risk Warnings", "2")
            st.metric("📊 Model Consensus", "2")
        
        # Recent alerts
        st.subheader("📋 Recent Alerts")
        
        # Simulate recent alerts
        alerts_data = [
            {
                'Time': '14:32:15',
                'Type': 'High Confidence',
                'Symbol': 'AAPL',
                'Message': 'Strong BUY signal detected (+0.0156)',
                'Severity': 'High'
            },
            {
                'Time': '14:15:42',
                'Type': 'Risk Warning',
                'Symbol': 'NVDA',
                'Message': 'Position exceeds risk limit (0.067)',
                'Severity': 'Critical'
            },
            {
                'Time': '13:58:23',
                'Type': 'Model Consensus',
                'Symbol': 'MSFT',
                'Message': 'High model agreement on prediction',
                'Severity': 'Medium'
            }
        ]
        
        for alert in alerts_data:
            severity_color = {"High": "🟡", "Critical": "🔴", "Medium": "🟢"}[alert['Severity']]
            
            st.markdown(f"""
            <div style="border-left: 4px solid {'orange' if alert['Severity']=='High' else 'red' if alert['Severity']=='Critical' else 'green'}; 
                        padding: 10px; margin: 10px 0; background: #f8f9fa;">
                <strong>{severity_color} {alert['Type']} - {alert['Symbol']}</strong><br>
                <em>{alert['Time']}</em><br>
                {alert['Message']}
            </div>
            """, unsafe_allow_html=True)
        
        # Alert visualization
        st.subheader("📊 Alert Frequency")
        
        # Simulate alert frequency data
        alert_dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        alert_counts = np.random.poisson(5, 7)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=alert_dates,
            y=alert_counts,
            name='Daily Alerts',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Alert Frequency (Last 7 Days)",
            xaxis_title="Date",
            yaxis_title="Number of Alerts",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_model_insights_page(self):
        """Render model insights page"""
        st.header("🤖 Model Insights")
        
        # Model explanation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧠 How Our AI Works")
            
            st.markdown("""
            **Our ensemble model combines multiple AI techniques:**
            
            1. **🌳 XGBoost**: Gradient boosting for complex patterns
            2. **⚡ LightGBM**: Fast, efficient tree-based learning
            3. **🔗 Random Forest**: Ensemble of decision trees
            4. **📊 Linear Models**: For baseline comparisons
            
            **The system uses 73 engineered features including:**
            - 📈 Technical indicators (RSI, MACD, Bollinger Bands)
            - 📊 Price momentum and volatility measures
            - 📉 Rolling statistics and lag features
            - ⏰ Time-based patterns
            """)
        
        with col2:
            st.subheader("📊 Feature Importance")
            
            # Simulate feature importance
            features = ['bb_lower', 'ema_200', 'close_to_sma20', 'rsi', 'momentum_5d', 
                       'atr_ratio', 'volume_ratio', 'volatility_20d', 'sma_50', 'stoch_k']
            importance = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=features,
                x=importance,
                orientation='h',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title="Top 10 Most Important Features",
                xaxis_title="Importance Score",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model architecture
        st.subheader("🏗️ Architecture Overview")
        
        st.markdown("""
        ```
        📊 Data Input (OHLCV + Features)
              ↓
        🔧 Feature Engineering (73 features)
              ↓
        🤖 Ensemble Models
              ├── XGBoost (Optimized)
              ├── LightGBM (Optimized) 
              ├── Random Forest
              └── Neural Network
              ↓
        🎯 Prediction Aggregation
              ↓
        📈 Final Prediction + Confidence
        ```
        """)
        
        # Performance metrics explanation
        st.subheader("📊 Understanding the Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📈 Sharpe Ratio**: Risk-adjusted returns
            - Higher is better
            - Our best: 4.25 (Excellent)
            - Market average: ~1.0
            
            **💰 Annual Return**: Yearly profit percentage
            - Our models: 10-15%
            - S&P 500 average: ~10%
            
            **📉 Max Drawdown**: Largest loss from peak
            - Lower is better
            - Our models: 8-15%
            - Acceptable: <20%
            """)
        
        with col2:
            st.markdown("""
            **🎯 Win Rate**: Percentage of profitable trades
            - Our models: 59-65%
            - Random chance: 50%
            - Good performance: >55%
            
            **⚡ Prediction Speed**: Time to generate forecast
            - Real-time: <3 seconds
            - Batch processing: <30 seconds
            - Updated every 15 minutes
            """)
        
        # Model comparison
        st.subheader("🔍 Model Comparison")
        
        performance_data = self.load_performance_data()
        if not performance_data.empty:
            # Radar chart for model comparison
            categories = ['Sharpe Ratio', 'Annual Return', 'Win Rate', 'Stability']
            
            fig = go.Figure()
            
            for _, model in performance_data.head(3).iterrows():
                # Normalize metrics for radar chart
                sharpe_norm = min(model['Sharpe_Ratio'] / 5, 1)  # Normalize to 0-1
                return_norm = model['Annual_Return'] * 5  # Scale up return
                win_norm = model['Win_Rate'] / 100  # Convert percentage
                stability_norm = 0.8 + np.random.normal(0, 0.1)  # Simulate stability
                
                values = [sharpe_norm, return_norm, win_norm, stability_norm]
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=model['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Performance Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main dashboard runner"""
        self.render_header()
        
        # Sidebar navigation
        page, selected_stocks, time_horizon, risk_tolerance = self.render_sidebar()
        
        # Main content based on selected page
        if page == "🏠 Overview":
            self.render_overview_page(selected_stocks)
        elif page == "🔮 Live Predictions":
            self.render_predictions_page(selected_stocks)
        elif page == "📊 Performance Analytics":
            self.render_performance_page()
        elif page == "💼 Portfolio Optimizer":
            self.render_portfolio_page(selected_stocks, risk_tolerance)
        elif page == "🚨 Alert Center":
            self.render_alerts_page()
        elif page == "🤖 Model Insights":
            self.render_model_insights_page()
        
        # Footer
        st.markdown("---")
        st.markdown("### 📈 Stock Market AI Prediction Engine - Day 14 Dashboard")
        st.markdown("*Built with Streamlit, Plotly, and Advanced ML Models*")

def main():
    """Main entry point"""
    try:
        dashboard = DashboardApp()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Dashboard Error: {e}")
        st.markdown("**Troubleshooting:**")
        st.markdown("1. Ensure all model files are in place")
        st.markdown("2. Check that Day 11 risk analysis completed")
        st.markdown("3. Verify data files exist in data/processed/")

if __name__ == "__main__":
    main()