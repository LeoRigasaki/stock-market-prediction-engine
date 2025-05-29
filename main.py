#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 5
Advanced Exploratory Data Analysis & Market Pattern Recognition
"""

from src.config import Config
from src.market_analyzer import MarketAnalyzer
from loguru import logger
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def run_comprehensive_market_analysis():
    """Run comprehensive market analysis and pattern recognition"""
    print("\n🔬 Starting Advanced Market Analysis and Pattern Recognition...")
    
    # Initialize analyzer
    analyzer = MarketAnalyzer()
    
    # Load feature data from Day 4
    print("\n📊 Loading engineered features from Day 4...")
    df = analyzer.load_feature_data()
    if df.empty:
        print("❌ Failed to load feature data. Please run Day 4 first.")
        return None
    
    print(f"✅ Loaded data: {len(df):,} records, {df.shape[1]} features, {df['Ticker'].nunique()} stocks")
    print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # 1. Return Distribution Analysis
    print("\n📈 Analyzing return distributions and statistical properties...")
    return_analysis = analyzer.analyze_return_distributions(df)
    
    if return_analysis:
        print("✅ Return Distribution Analysis Results:")
        for return_type, stats in list(return_analysis.items())[:3]:  # Show first 3
            print(f"   {return_type}:")
            print(f"     Mean: {stats.get('mean', 0):.4f}%, Std: {stats.get('std', 0):.4f}%")
            print(f"     Skewness: {stats.get('skewness', 0):.4f}, Kurtosis: {stats.get('kurtosis', 0):.4f}")
            if 'is_normal_jb' in stats:
                print(f"     Normal Distribution (Jarque-Bera): {'Yes' if stats['is_normal_jb'] else 'No'}")
    
    # 2. Market Regime Detection
    print("\n🎯 Detecting market regimes using volatility and return patterns...")
    regime_df = analyzer.detect_market_regimes(df)
    
    if 'Market_Regime' in regime_df.columns:
        regime_counts = regime_df['Market_Regime'].value_counts()
        print("✅ Market Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = (count / len(regime_df)) * 100
            print(f"   {regime}: {count:,} observations ({pct:.1f}%)")
    
    # 3. Stock Correlation Analysis
    print("\n🔗 Analyzing inter-stock correlations and market relationships...")
    correlation_matrix, beta_analysis = analyzer.analyze_stock_correlations(regime_df)
    
    if beta_analysis:
        print("✅ Beta Analysis Results:")
        beta_df = pd.DataFrame(beta_analysis).T.sort_values('beta', ascending=False)
        print("   Top 5 Highest Beta Stocks:")
        for idx, (stock, data) in enumerate(beta_df.head().iterrows(), 1):
            print(f"     {idx}. {stock}: Beta={data['beta']:.3f}, R²={data['r_squared']:.3f}")
    
    # 4. Principal Component Analysis
    print("\n🔬 Performing Principal Component Analysis on features...")
    pca, pca_df, feature_importance = analyzer.perform_pca_analysis(regime_df)
    
    if pca:
        variance_explained = pca.explained_variance_ratio_.cumsum()
        print("✅ PCA Analysis Results:")
        print(f"   First 5 components explain {variance_explained[4]:.1%} of variance")
        print("   Top 5 Most Important Features for PC1:")
        for idx, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
            print(f"     {idx}. {row['Feature']}: {row['PC1_Loading']:.4f}")
    
    # 5. Anomaly Detection
    print("\n🚨 Detecting anomalies and outliers in stock data...")
    anomaly_df = analyzer.detect_anomalies(regime_df)
    
    if 'is_anomaly' in anomaly_df.columns:
        anomaly_count = anomaly_df['is_anomaly'].sum()
        anomaly_pct = (anomaly_count / len(anomaly_df)) * 100
        print(f"✅ Anomaly Detection Results:")
        print(f"   Detected {anomaly_count:,} anomalies ({anomaly_pct:.2f}% of data)")
        
        # Show top anomalous events by stock
        anomalies_by_stock = anomaly_df[anomaly_df['is_anomaly']].groupby('Ticker').size().sort_values(ascending=False)
        print("   Top 5 Stocks with Most Anomalies:")
        for idx, (stock, count) in enumerate(anomalies_by_stock.head().items(), 1):
            print(f"     {idx}. {stock}: {count} anomalies")
    
    # 6. Seasonal Pattern Analysis
    print("\n📅 Analyzing seasonal and cyclical patterns...")
    seasonal_analysis = analyzer.analyze_seasonal_patterns(anomaly_df)
    
    if seasonal_analysis:
        print("✅ Seasonal Pattern Analysis:")
        
        # Monthly patterns
        if 'monthly_patterns' in seasonal_analysis:
            monthly_data = seasonal_analysis['monthly_patterns']
            best_month = monthly_data.loc[monthly_data['mean'].idxmax()]
            worst_month = monthly_data.loc[monthly_data['mean'].idxmin()]
            print(f"   Best performing month: {best_month['Month_Name']} ({best_month['mean']:.3f}% avg return)")
            print(f"   Worst performing month: {worst_month['Month_Name']} ({worst_month['mean']:.3f}% avg return)")
        
        # Day of week patterns
        if 'day_of_week_patterns' in seasonal_analysis:
            dow_data = seasonal_analysis['day_of_week_patterns']
            best_day = dow_data.loc[dow_data['mean'].idxmax()]
            worst_day = dow_data.loc[dow_data['mean'].idxmin()]
            print(f"   Best performing day: {best_day['Day_Name']} ({best_day['mean']:.3f}% avg return)")
            print(f"   Worst performing day: {worst_day['Day_Name']} ({worst_day['mean']:.3f}% avg return)")
        
        # Statistical significance
        if 'monthly_anova' in seasonal_analysis:
            anova_result = seasonal_analysis['monthly_anova']
            significance = "significant" if anova_result['significant'] else "not significant"
            print(f"   Monthly differences are {significance} (p-value: {anova_result['p_value']:.4f})")
    
    # 7. Create Interactive Dashboard
    print("\n📊 Creating comprehensive interactive analysis dashboard...")
    try:
        dashboard_fig = analyzer.create_interactive_dashboard(
            anomaly_df, correlation_matrix, pca_df, seasonal_analysis
        )
        
        # Save dashboard
        plots_dir = Config.PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        dashboard_path = plots_dir / "day5_interactive_dashboard.html"
        dashboard_fig.write_html(str(dashboard_path))
        print(f"✅ Interactive dashboard saved: {dashboard_path}")
    except Exception as e:
        print(f"⚠️ Dashboard creation failed: {e}")
    
    # 8. Save All Analysis Results
    print("\n💾 Saving comprehensive analysis results...")
    saved_files = analyzer.save_analysis_results(
        return_analysis, correlation_matrix, pca_df, seasonal_analysis, anomaly_df
    )
    
    print("✅ Analysis results saved:")
    for analysis_type, path in saved_files.items():
        if path:
            print(f"   {analysis_type}: {path}")
    
    return {
        'processed_data': anomaly_df,
        'return_analysis': return_analysis,
        'correlation_matrix': correlation_matrix,
        'pca_results': (pca, pca_df, feature_importance),
        'seasonal_analysis': seasonal_analysis,
        'saved_files': saved_files
    }

def generate_market_insights(results):
    """Generate actionable market insights from analysis"""
    print("\n🎯 Generating Actionable Market Insights...")
    
    insights = []
    
    # Return distribution insights
    if results['return_analysis']:
        for return_type, stats in results['return_analysis'].items():
            if 'return_5d' in return_type:
                if stats.get('skewness', 0) > 0.5:
                    insights.append(f"📈 {return_type} shows positive skew - more frequent small losses, occasional large gains")
                elif stats.get('skewness', 0) < -0.5:
                    insights.append(f"📉 {return_type} shows negative skew - more frequent small gains, occasional large losses")
                
                if stats.get('kurtosis', 0) > 3:
                    insights.append(f"⚡ {return_type} shows fat tails - extreme events more likely than normal distribution")
    
    # Correlation insights
    if not results['correlation_matrix'].empty:
        corr_matrix = results['correlation_matrix']
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        if avg_correlation > 0.7:
            insights.append("🔗 High inter-stock correlation - diversification benefits limited")
        elif avg_correlation < 0.3:
            insights.append("🎯 Low inter-stock correlation - good diversification opportunities")
        else:
            insights.append("⚖️ Moderate inter-stock correlation - balanced portfolio risk")
    
    # Seasonal insights
    if results['seasonal_analysis'] and 'monthly_anova' in results['seasonal_analysis']:
        if results['seasonal_analysis']['monthly_anova']['significant']:
            insights.append("📅 Significant seasonal patterns detected - timing strategies may be effective")
        else:
            insights.append("🎲 No significant seasonal patterns - market timing unlikely to add value")
    
    # PCA insights
    if results['pca_results'][0]:  # pca object exists
        pca = results['pca_results'][0]
        if pca.explained_variance_ratio_[0] > 0.3:
            insights.append("🎯 Market dominated by single factor - systematic risk high")
        else:
            insights.append("🌐 Market driven by multiple factors - diversified risk sources")
    
    print("🔍 Key Market Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    return insights

def main():
    """Main execution function for Day 5"""
    
    print("🚀 Stock Market Prediction Engine - Day 5")
    print("Advanced Exploratory Data Analysis & Market Pattern Recognition")
    print("=" * 70)
    
    # Check dependencies from previous days
    config = Config()
    features_path = config.FEATURES_DATA_PATH / "selected_features.csv"
    
    if not features_path.exists():
        print("\n❌ Feature dataset not found!")
        print("Please run Day 4 first to generate engineered features")
        print("Command: python main.py  # (from Day 4)")
        return
    
    # Run comprehensive analysis
    results = run_comprehensive_market_analysis()
    
    if results is None:
        print("\n❌ Market analysis failed!")
        return
    
    # Generate actionable insights
    insights = generate_market_insights(results)
    
    # Display final results
    processed_df = results['processed_data']
    
    print("\n🎯 Day 5 Completed Successfully!")
    print("=" * 70)
    print("✅ Advanced exploratory data analysis completed")
    print("✅ Market regime detection performed")
    print("✅ Statistical hypothesis testing conducted")
    print("✅ Principal component analysis executed")
    print("✅ Anomaly detection implemented")
    print("✅ Seasonal pattern analysis completed")
    print("✅ Interactive dashboard created")
    print("✅ Comprehensive analysis results saved")
    
    print(f"\n📊 Final Analysis Summary:")
    print(f"   Records analyzed: {len(processed_df):,}")
    print(f"   Stocks analyzed: {processed_df['Ticker'].nunique()}")
    print(f"   Features analyzed: {processed_df.shape[1]}")
    print(f"   Date range: {processed_df['Date'].min().date()} to {processed_df['Date'].max().date()}")
    print(f"   Market insights generated: {len(insights)}")
    
    if 'Market_Regime' in processed_df.columns:
        regime_counts = processed_df['Market_Regime'].value_counts()
        print(f"\n🎯 Market Regime Summary:")
        for regime, count in regime_counts.head(3).items():
            pct = (count / len(processed_df)) * 100
            print(f"   {regime}: {pct:.1f}% of observations")
    
    if 'is_anomaly' in processed_df.columns:
        anomaly_count = processed_df['is_anomaly'].sum()
        anomaly_pct = (anomaly_count / len(processed_df)) * 100
        print(f"\n🚨 Anomaly Detection Summary:")
        print(f"   Anomalies detected: {anomaly_count:,} ({anomaly_pct:.2f}%)")
    
    print("\n📋 Ready for Day 6:")
    print("1. Baseline machine learning model development")
    print("2. Model evaluation and validation framework")
    print("3. Feature importance analysis from ML perspective")
    print("4. Cross-validation with time series considerations")

if __name__ == "__main__":
    main()