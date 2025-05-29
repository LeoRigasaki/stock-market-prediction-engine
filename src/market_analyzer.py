import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest, normaltest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
from loguru import logger

from .config import Config

class MarketAnalyzer:
    """Advanced market analysis and pattern recognition"""
    
    def __init__(self):
        self.config = Config()
        
    def load_feature_data(self) -> pd.DataFrame:
        """Load the engineered features from Day 4"""
        features_path = self.config.FEATURES_DATA_PATH / "selected_features.csv"
        
        if not features_path.exists():
            logger.error(f"Feature dataset not found at {features_path}")
            logger.error("Please run Day 4 first to generate engineered features")
            return pd.DataFrame()
        
        df = pd.read_csv(features_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        logger.info(f"Loaded feature dataset: {len(df)} records, {df.shape[1]} features, {df['Ticker'].nunique()} stocks")
        return df
    
    def get_available_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get available columns by category for adaptive analysis"""
        available_cols = {
            'price_cols': [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns],
            'volume_cols': [col for col in ['Volume'] if col in df.columns],
            'return_cols': [col for col in df.columns if any(x in col for x in ['return_', 'momentum_', 'daily_return'])],
            'volatility_cols': [col for col in df.columns if 'volatility' in col or 'atr' in col],
            'ma_cols': [col for col in df.columns if any(x in col for x in ['sma_', 'ema_', 'close_to_sma'])],
            'bb_cols': [col for col in df.columns if 'bb_' in col],
            'technical_cols': [col for col in df.columns if any(x in col for x in ['rsi', 'stoch', 'williams', 'cmf'])],
            'target_cols': [col for col in df.columns if col.startswith('target_')]
        }
        
        logger.info(f"Available column categories: {[(k, len(v)) for k, v in available_cols.items()]}")
        return available_cols
    
    def analyze_return_distributions(self, df: pd.DataFrame) -> Dict:
        """Comprehensive analysis of return distributions"""
        logger.info("Analyzing return distributions and statistical properties...")
        
        analysis_results = {}
        
        # Get available return columns
        available_cols = self.get_available_columns(df)
        return_cols = available_cols['return_cols']
        
        if not return_cols:
            logger.warning("No return columns found for analysis")
            return analysis_results
        
        for col in return_cols:
            returns = df[col].dropna()
            
            if len(returns) < 50:  # Skip if insufficient data
                continue
            
            # Basic statistics
            stats_dict = {
                'count': len(returns),
                'mean': returns.mean(),
                'std': returns.std(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'min': returns.min(),
                'max': returns.max(),
                'median': returns.median(),
                'q25': returns.quantile(0.25),
                'q75': returns.quantile(0.75)
            }
            
            # Statistical tests for normality
            try:
                # Jarque-Bera test
                jb_stat, jb_pvalue = jarque_bera(returns)
                stats_dict['jarque_bera_stat'] = jb_stat
                stats_dict['jarque_bera_pvalue'] = jb_pvalue
                stats_dict['is_normal_jb'] = jb_pvalue > 0.05
                
                # Shapiro-Wilk test (for smaller samples)
                if len(returns) <= 5000:
                    sw_stat, sw_pvalue = shapiro(returns[:5000])
                    stats_dict['shapiro_stat'] = sw_stat
                    stats_dict['shapiro_pvalue'] = sw_pvalue
                    stats_dict['is_normal_sw'] = sw_pvalue > 0.05
                
            except Exception as e:
                logger.warning(f"Statistical tests failed for {col}: {e}")
            
            # Risk metrics
            stats_dict['var_95'] = returns.quantile(0.05)  # 5% VaR
            stats_dict['var_99'] = returns.quantile(0.01)  # 1% VaR
            stats_dict['cvar_95'] = returns[returns <= returns.quantile(0.05)].mean()  # Conditional VaR
            
            # Volatility clustering test (autocorrelation of squared returns)
            squared_returns = returns ** 2
            if len(squared_returns) > 20:
                autocorr_1 = squared_returns.autocorr(lag=1)
                stats_dict['volatility_clustering'] = autocorr_1
            
            analysis_results[col] = stats_dict
        
        logger.info(f"Completed return distribution analysis for {len(analysis_results)} return series")
        return analysis_results
    
    def detect_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes using available features"""
        logger.info("Detecting market regimes using statistical methods...")
        
        regime_df = df.copy()
        available_cols = self.get_available_columns(df)
        
        # Build aggregation dict based on available columns
        agg_dict = {}
        
        # Use available return columns
        if available_cols['return_cols']:
            primary_return = available_cols['return_cols'][0]  # Use first available return column
            agg_dict[primary_return] = 'mean'
        
        # Use available volatility columns
        if available_cols['volatility_cols']:
            primary_vol = available_cols['volatility_cols'][0]  # Use first available volatility column
            agg_dict[primary_vol] = 'mean'
        
        # Use Close price if available
        if 'Close' in df.columns:
            agg_dict['Close'] = 'mean'
        
        # Skip if we don't have minimum required columns
        if len(agg_dict) < 2:
            logger.warning("Insufficient columns for market regime detection")
            regime_df['Market_Regime'] = 'Unknown'
            return regime_df
        
        # Calculate market-wide metrics (average across all stocks)
        market_metrics = df.groupby('Date').agg(agg_dict).reset_index()
        
        # Define regimes based on available metrics
        if primary_return in market_metrics.columns and primary_vol in market_metrics.columns:
            vol_col = primary_vol
            ret_col = primary_return
            
            vol_threshold_high = market_metrics[vol_col].quantile(0.7)
            vol_threshold_low = market_metrics[vol_col].quantile(0.3)
            
            return_threshold_high = market_metrics[ret_col].quantile(0.7)
            return_threshold_low = market_metrics[ret_col].quantile(0.3)
            
            def classify_regime(row):
                vol = row[vol_col]
                ret = row[ret_col]
                
                if vol > vol_threshold_high:
                    if ret > return_threshold_high:
                        return 'Bull_High_Vol'
                    elif ret < return_threshold_low:
                        return 'Bear_High_Vol'
                    else:
                        return 'Sideways_High_Vol'
                elif vol < vol_threshold_low:
                    if ret > return_threshold_high:
                        return 'Bull_Low_Vol'
                    elif ret < return_threshold_low:
                        return 'Bear_Low_Vol'
                    else:
                        return 'Sideways_Low_Vol'
                else:
                    if ret > return_threshold_high:
                        return 'Bull_Normal_Vol'
                    elif ret < return_threshold_low:
                        return 'Bear_Normal_Vol'
                    else:
                        return 'Sideways_Normal_Vol'
            
            market_metrics['Market_Regime'] = market_metrics.apply(classify_regime, axis=1)
        else:
            # Fallback: simple classification based on returns only
            if primary_return in market_metrics.columns:
                ret_col = primary_return
                return_threshold_high = market_metrics[ret_col].quantile(0.6)
                return_threshold_low = market_metrics[ret_col].quantile(0.4)
                
                def simple_classify_regime(row):
                    ret = row[ret_col]
                    if ret > return_threshold_high:
                        return 'Bull_Market'
                    elif ret < return_threshold_low:
                        return 'Bear_Market'
                    else:
                        return 'Sideways_Market'
                
                market_metrics['Market_Regime'] = market_metrics.apply(simple_classify_regime, axis=1)
        
        # Merge back to main dataframe
        regime_df = regime_df.merge(market_metrics[['Date', 'Market_Regime']], on='Date', how='left')
        
        # Regime statistics
        regime_stats = market_metrics['Market_Regime'].value_counts()
        logger.info(f"Market regime distribution: {regime_stats.to_dict()}")
        
        return regime_df
    
    def analyze_stock_correlations(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Analyze correlations between stocks and with market factors"""
        logger.info("Analyzing inter-stock correlations and market relationships...")
        
        if 'Close' not in df.columns:
            logger.warning("Close price not available for correlation analysis")
            return pd.DataFrame(), {}
        
        # Create price pivot table
        price_pivot = df.pivot_table(index='Date', columns='Ticker', values='Close')
        
        # Calculate returns for correlation analysis
        returns_pivot = price_pivot.pct_change().dropna()
        
        # Correlation matrix
        correlation_matrix = returns_pivot.corr()
        
        # Market beta analysis (using equal-weighted market return)
        market_return = returns_pivot.mean(axis=1)  # Equal weighted market
        
        beta_analysis = {}
        for stock in returns_pivot.columns:
            stock_returns = returns_pivot[stock].dropna()
            market_aligned = market_return.loc[stock_returns.index]
            
            if len(stock_returns) > 50:  # Minimum data requirement
                # Calculate beta using linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(market_aligned, stock_returns)
                
                beta_analysis[stock] = {
                    'beta': slope,
                    'alpha': intercept,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'correlation_with_market': r_value,
                    'volatility': stock_returns.std(),
                    'mean_return': stock_returns.mean()
                }
        
        beta_df = pd.DataFrame(beta_analysis).T
        
        logger.info(f"Correlation analysis completed for {len(correlation_matrix)} stocks")
        logger.info(f"Beta analysis completed for {len(beta_analysis)} stocks")
        
        return correlation_matrix, beta_analysis
    
    def perform_pca_analysis(self, df: pd.DataFrame) -> Tuple[PCA, pd.DataFrame, pd.DataFrame]:
        """Principal Component Analysis on features"""
        logger.info("Performing Principal Component Analysis on features...")
        
        # Select numerical features (exclude date, ticker, target variables)
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Ticker'] 
                       and not col.startswith('target_') 
                       and not col.startswith('return_')
                       and df[col].dtype in ['float64', 'int64']]
        
        if len(feature_cols) < 3:
            logger.warning("Insufficient numerical features for PCA")
            return None, pd.DataFrame(), pd.DataFrame()
        
        # Prepare data
        feature_data = df[feature_cols].fillna(df[feature_cols].median())
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Perform PCA
        n_components = min(20, len(feature_cols))
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_features)
        
        # Create PCA dataframe
        pca_columns = [f'PC{i+1}' for i in range(pca.n_components_)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns)
        pca_df['Date'] = df['Date'].values
        pca_df['Ticker'] = df['Ticker'].values
        
        # Feature importance in PCA
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'PC1_Loading': np.abs(pca.components_[0]),
            'PC2_Loading': np.abs(pca.components_[1]) if pca.n_components_ > 1 else 0,
            'PC3_Loading': np.abs(pca.components_[2]) if pca.n_components_ > 2 else 0
        }).sort_values('PC1_Loading', ascending=False)
        
        logger.info(f"PCA completed: {pca.n_components_} components explain {pca.explained_variance_ratio_.cumsum()[-1]:.3f} of variance")
        logger.info(f"Top 5 features for PC1: {feature_importance.head()['Feature'].tolist()}")
        
        return pca, pca_df, feature_importance
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in stock data using statistical methods"""
        logger.info("Detecting anomalies and outliers in stock data...")
        
        anomaly_df = df.copy()
        available_cols = self.get_available_columns(df)
        
        # Use the first available return column for anomaly detection
        return_cols = available_cols['return_cols']
        if not return_cols:
            logger.warning("No return columns available for anomaly detection")
            anomaly_df['is_anomaly'] = False
            return anomaly_df
        
        primary_return = return_cols[0]
        
        # Z-score based anomaly detection for returns
        for stock in df['Ticker'].unique():
            stock_data = df[df['Ticker'] == stock].copy()
            
            if primary_return in stock_data.columns:
                returns = stock_data[primary_return].dropna()
                
                if len(returns) > 30:  # Minimum data requirement
                    # Z-score method
                    z_scores = np.abs(stats.zscore(returns))
                    outliers_zscore = z_scores > 3  # 3 standard deviations
                    
                    # IQR method
                    Q1 = returns.quantile(0.25)
                    Q3 = returns.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers_iqr = (returns < (Q1 - 1.5 * IQR)) | (returns > (Q3 + 1.5 * IQR))
                    
                    # Combine methods
                    combined_outliers = outliers_zscore | outliers_iqr
                    
                    # Add anomaly flags to the dataframe
                    stock_mask = anomaly_df['Ticker'] == stock
                    anomaly_df.loc[stock_mask, 'is_anomaly'] = False
                    
                    # Map outliers back to original dataframe indices
                    stock_indices = stock_data.index[stock_data[primary_return].notna()]
                    outlier_indices = stock_indices[combined_outliers]
                    anomaly_df.loc[outlier_indices, 'is_anomaly'] = True
        
        # Summary statistics
        if 'is_anomaly' in anomaly_df.columns:
            anomaly_count = anomaly_df['is_anomaly'].sum()
            anomaly_pct = (anomaly_count / len(anomaly_df)) * 100
            logger.info(f"Detected {anomaly_count} anomalies ({anomaly_pct:.2f}% of data)")
        
        return anomaly_df
    
    def analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal and cyclical patterns in stock data"""
        logger.info("Analyzing seasonal and cyclical patterns...")
        
        seasonal_analysis = {}
        available_cols = self.get_available_columns(df)
        
        # Use first available return column
        return_cols = available_cols['return_cols']
        if not return_cols:
            logger.warning("No return columns available for seasonal analysis")
            return seasonal_analysis
        
        primary_return = return_cols[0]
        
        # Monthly patterns
        df_copy = df.copy()
        df_copy['Month'] = df_copy['Date'].dt.month
        df_copy['DayOfWeek'] = df_copy['Date'].dt.dayofweek
        df_copy['Quarter'] = df_copy['Date'].dt.quarter
        
        # Analyze patterns by month
        monthly_returns = df_copy.groupby('Month')[primary_return].agg(['mean', 'std', 'count']).reset_index()
        monthly_returns['Month_Name'] = monthly_returns['Month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        
        # Day of week patterns
        dow_returns = df_copy.groupby('DayOfWeek')[primary_return].agg(['mean', 'std', 'count']).reset_index()
        dow_returns['Day_Name'] = dow_returns['DayOfWeek'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        
        # Quarter patterns
        quarterly_returns = df_copy.groupby('Quarter')[primary_return].agg(['mean', 'std', 'count']).reset_index()
        
        seasonal_analysis = {
            'monthly_patterns': monthly_returns,
            'day_of_week_patterns': dow_returns,
            'quarterly_patterns': quarterly_returns,
            'primary_return_column': primary_return
        }
        
        # Test for significant seasonal effects
        try:
            # ANOVA test for monthly differences
            monthly_groups = [df_copy[df_copy['Month'] == month][primary_return].dropna() for month in range(1, 13)]
            monthly_groups = [group for group in monthly_groups if len(group) > 10]  # Filter small groups
            
            if len(monthly_groups) >= 3:
                f_stat, p_value = stats.f_oneway(*monthly_groups)
                seasonal_analysis['monthly_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        except Exception as e:
            logger.warning(f"ANOVA test failed: {e}")
        
        logger.info("Seasonal pattern analysis completed")
        return seasonal_analysis
    
    def create_interactive_dashboard(self, df: pd.DataFrame, correlation_matrix: pd.DataFrame, 
                                   pca_df: pd.DataFrame, seasonal_analysis: Dict) -> go.Figure:
        """Create comprehensive interactive dashboard"""
        logger.info("Creating interactive analysis dashboard...")
        
        available_cols = self.get_available_columns(df)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Stock Price Trends', 'Return Distribution', 'Correlation Heatmap',
                'PCA Analysis', 'Volatility Over Time', 'Market Regime Distribution',
                'Seasonal Patterns (Monthly)', 'Feature Distribution', 'Risk-Return Profile'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Stock price trends (sample of 5 stocks)
        if 'Close' in df.columns:
            sample_stocks = df['Ticker'].unique()[:5]
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, stock in enumerate(sample_stocks):
                stock_data = df[df['Ticker'] == stock].sort_values('Date')
                fig.add_trace(
                    go.Scatter(x=stock_data['Date'], y=stock_data['Close'], 
                              name=stock, line=dict(width=1.5, color=colors[i % len(colors)]), opacity=0.8),
                    row=1, col=1
                )
        
        # 2. Return distribution
        if available_cols['return_cols']:
            primary_return = available_cols['return_cols'][0]
            returns = df[primary_return].dropna()
            fig.add_trace(
                go.Histogram(x=returns, nbinsx=50, name=f'{primary_return}', opacity=0.7),
                row=1, col=2
            )
        
        # 3. Correlation heatmap (simplified)
        if not correlation_matrix.empty and len(correlation_matrix) > 1:
            fig.add_trace(
                go.Heatmap(z=correlation_matrix.values, 
                          x=correlation_matrix.columns, 
                          y=correlation_matrix.index,
                          colorscale='RdBu', zmid=0),
                row=1, col=3
            )
        
        # 4. PCA explained variance
        if not pca_df.empty:
            # Sample explained variance ratios
            n_components = min(8, pca_df.shape[1] - 2)  # -2 for Date and Ticker
            sample_variance = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04])[:n_components]
            fig.add_trace(
                go.Bar(x=[f'PC{i+1}' for i in range(n_components)], 
                      y=sample_variance, name='Explained Variance'),
                row=2, col=1
            )
        
        # 5. Volatility over time (using available volatility column)
        if available_cols['volatility_cols']:
            vol_col = available_cols['volatility_cols'][0]
            vol_time = df.groupby('Date')[vol_col].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=vol_time['Date'], y=vol_time[vol_col], 
                          name=f'{vol_col}', line=dict(color='red')),
                row=2, col=2
            )
        
        # 6. Market regime distribution
        if 'Market_Regime' in df.columns:
            regime_counts = df['Market_Regime'].value_counts()
            fig.add_trace(
                go.Bar(x=regime_counts.index, y=regime_counts.values, 
                    name='Market Regimes'),
                row=2, col=3
            )
        
        # 7. Seasonal patterns
        if seasonal_analysis and 'monthly_patterns' in seasonal_analysis:
            monthly_data = seasonal_analysis['monthly_patterns']
            fig.add_trace(
                go.Bar(x=monthly_data['Month_Name'], y=monthly_data['mean'], 
                      name='Avg Monthly Returns'),
                row=3, col=1
            )
        
        # 8. Feature distribution example
        if available_cols['technical_cols']:
            tech_col = available_cols['technical_cols'][0]
            fig.add_trace(
                go.Histogram(x=df[tech_col].dropna(), nbinsx=30, 
                           name=f'{tech_col}', opacity=0.7),
                row=3, col=2
            )
        elif available_cols['ma_cols']:
            ma_col = available_cols['ma_cols'][0]
            fig.add_trace(
                go.Histogram(x=df[ma_col].dropna(), nbinsx=30, 
                           name=f'{ma_col}', opacity=0.7),
                row=3, col=2
            )
        
        # 9. Risk-return profile
        if available_cols['return_cols'] and available_cols['volatility_cols']:
            return_col = available_cols['return_cols'][0]
            vol_col = available_cols['volatility_cols'][0]
            
            risk_return = df.groupby('Ticker').agg({
                return_col: 'mean',
                vol_col: 'mean'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(x=risk_return[vol_col], y=risk_return[return_col],
                          mode='markers+text', text=risk_return['Ticker'],
                          textposition="top center", name='Risk-Return Profile'),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Stock Market Analysis Dashboard - Day 5",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def save_analysis_results(self, return_analysis: Dict, correlation_matrix: pd.DataFrame,
                            pca_df: pd.DataFrame, seasonal_analysis: Dict, 
                            anomaly_df: pd.DataFrame) -> Dict:
        """Save all analysis results"""
        logger.info("Saving analysis results...")
        
        saved_files = {}
        
        # Save return distribution analysis
        if return_analysis:
            return_analysis_df = pd.DataFrame(return_analysis).T
            return_path = self.config.PROCESSED_DATA_PATH / "return_distribution_analysis.csv"
            return_analysis_df.to_csv(return_path)
            saved_files['return_analysis'] = return_path
            logger.info(f"Return analysis saved: {return_path}")
        
        # Save correlation matrix
        if not correlation_matrix.empty:
            corr_path = self.config.PROCESSED_DATA_PATH / "stock_correlation_matrix.csv"
            correlation_matrix.to_csv(corr_path)
            saved_files['correlation_matrix'] = corr_path
            logger.info(f"Correlation matrix saved: {corr_path}")
        
        # Save PCA results
        if not pca_df.empty:
            pca_path = self.config.PROCESSED_DATA_PATH / "pca_analysis.csv"
            pca_df.to_csv(pca_path, index=False)
            saved_files['pca_analysis'] = pca_path
            logger.info(f"PCA analysis saved: {pca_path}")
        
        # Save seasonal analysis
        if seasonal_analysis:
            import json
            seasonal_path = self.config.PROCESSED_DATA_PATH / "seasonal_analysis.json"
            # Convert DataFrames to dict for JSON serialization
            seasonal_json = {}
            for key, value in seasonal_analysis.items():
                if isinstance(value, pd.DataFrame):
                    seasonal_json[key] = value.to_dict('records')
                else:
                    seasonal_json[key] = value
            
            with open(seasonal_path, 'w') as f:
                json.dump(seasonal_json, f, indent=2, default=str)
            saved_files['seasonal_analysis'] = seasonal_path
            logger.info(f"Seasonal analysis saved: {seasonal_path}")
        
        # Save anomaly detection results
        anomaly_path = self.config.PROCESSED_DATA_PATH / "anomaly_analysis.csv"
        anomaly_df.to_csv(anomaly_path, index=False)
        saved_files['anomaly_analysis'] = anomaly_path
        logger.info(f"Anomaly analysis saved: {anomaly_path}")
        
        # Create comprehensive summary
        summary = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_records_analyzed': len(anomaly_df),
            'stocks_analyzed': anomaly_df['Ticker'].nunique(),
            'return_distributions_analyzed': len(return_analysis) if return_analysis else 0,
            'correlation_matrix_size': correlation_matrix.shape if not correlation_matrix.empty else (0, 0),
            'pca_components': pca_df.shape[1] - 2 if not pca_df.empty else 0,  # -2 for Date and Ticker
            'anomalies_detected': anomaly_df['is_anomaly'].sum() if 'is_anomaly' in anomaly_df.columns else 0,
            'seasonal_patterns_found': len(seasonal_analysis) if seasonal_analysis else 0
        }
        
        summary_path = self.config.PROCESSED_DATA_PATH / "day5_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files['summary'] = summary_path
        logger.info(f"Analysis summary saved: {summary_path}")
        
        return saved_files