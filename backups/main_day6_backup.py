#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 6
Baseline Machine Learning Model Development
"""

from src.config import Config
from src.ml_models import MLModelFramework
from loguru import logger
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def run_baseline_ml_modeling():
    """Run comprehensive baseline ML model development"""
    print("\n🤖 Starting Baseline Machine Learning Model Development...")
    
    # Initialize ML framework
    ml_framework = MLModelFramework()
    
    # Load feature data from Day 4
    print("\n📊 Loading engineered features from Day 4...")
    df = ml_framework.load_feature_data()
    if df.empty:
        print("❌ Failed to load feature data. Please run Day 4 first.")
        return None
    
    print(f"✅ Loaded data: {len(df):,} records, {df.shape[1]} features, {df['Ticker'].nunique()} stocks")
    print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Create time series splits for validation
    print("\n⏰ Creating time-series aware train/validation splits...")
    cv_splits = ml_framework.create_time_series_splits(df, n_splits=5)
    print(f"✅ Created {len(cv_splits)} time-series validation folds")
    
    # Prepare data for regression modeling
    print("\n📈 Preparing data for regression modeling (predicting returns)...")
    X_reg, y_reg, feature_cols = ml_framework.prepare_modeling_data(
        df, target_col='return_5d', prediction_type='regression'
    )
    print(f"✅ Regression data prepared: {X_reg.shape[0]} samples, {X_reg.shape[1]} features")
    print(f"   Target stats: mean={y_reg.mean():.4f}, std={y_reg.std():.4f}")
    
    # Train regression models
    print("\n🔧 Training baseline regression models...")
    regression_models = ml_framework.train_baseline_models(
        X_reg, y_reg, prediction_type='regression'
    )
    print(f"✅ Trained {len(regression_models)} regression models")
    
    # Evaluate regression models
    print("\n📊 Evaluating regression models with time-series cross-validation...")
    regression_results = ml_framework.evaluate_regression_models(
        regression_models, X_reg, y_reg, cv_splits
    )
    
    print("✅ Regression Model Performance:")
    for name, results in regression_results.items():
        print(f"   {name}:")
        print(f"     R² Score: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
        print(f"     RMSE: {results['rmse_mean']:.4f}")
        print(f"     MAE: {results['cv_mae_mean']:.4f} ± {results['cv_mae_std']:.4f}")
    
    # Prepare data for classification modeling
    print("\n🎯 Preparing data for classification modeling (predicting direction)...")
    X_class, y_class, _ = ml_framework.prepare_modeling_data(
        df, target_col='return_5d', prediction_type='classification'
    )
    print(f"✅ Classification data prepared: {X_class.shape[0]} samples, {X_class.shape[1]} features")
    print(f"   Class distribution: {np.bincount(y_class)} (Negative: {np.bincount(y_class)[0]}, Positive: {np.bincount(y_class)[1]})")
    
    # Train classification models
    print("\n🔧 Training baseline classification models...")
    classification_models = ml_framework.train_baseline_models(
        X_class, y_class, prediction_type='classification'
    )
    print(f"✅ Trained {len(classification_models)} classification models")
    
    # Evaluate classification models
    print("\n📊 Evaluating classification models with time-series cross-validation...")
    classification_results = ml_framework.evaluate_classification_models(
        classification_models, X_class, y_class, cv_splits
    )
    
    print("✅ Classification Model Performance:")
    for name, results in classification_results.items():
        print(f"   {name}:")
        print(f"     Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        print(f"     F1 Score: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
        print(f"     Precision: {results['cv_precision_mean']:.4f} ± {results['cv_precision_std']:.4f}")
        print(f"     Recall: {results['cv_recall_mean']:.4f} ± {results['cv_recall_std']:.4f}")
    
    # Feature importance analysis
    print("\n🔍 Analyzing feature importance from ML perspective...")
    
    # Display top features from Random Forest models
    if 'Random_Forest' in regression_results:
        rf_reg_importance = regression_results['Random_Forest'].get('feature_importance')
        if rf_reg_importance is not None:
            print("🌲 Top 10 Features (Random Forest Regression):")
            for idx, (_, row) in enumerate(rf_reg_importance.head(10).iterrows(), 1):
                print(f"   {idx:2d}. {row['feature']}: {row['importance']:.4f}")
    
    if 'Random_Forest' in classification_results:
        rf_class_importance = classification_results['Random_Forest'].get('feature_importance')
        if rf_class_importance is not None:
            print("🌲 Top 10 Features (Random Forest Classification):")
            for idx, (_, row) in enumerate(rf_class_importance.head(10).iterrows(), 1):
                print(f"   {idx:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Create comprehensive visualizations
    print("\n📊 Creating comprehensive model comparison visualizations...")
    try:
        comparison_fig = ml_framework.create_model_comparison_plots(
            regression_results, classification_results
        )
        
        # Save visualization
        plots_dir = Config.PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        comparison_path = plots_dir / "day6_model_comparison.html"
        comparison_fig.write_html(str(comparison_path))
        print(f"✅ Model comparison plots saved: {comparison_path}")
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
    
    # Save all models and results
    print("\n💾 Saving trained models and evaluation results...")
    saved_files = ml_framework.save_models_and_results(
        regression_results, classification_results, feature_cols
    )
    
    print("✅ Models and results saved:")
    for file_type, path in saved_files.items():
        print(f"   {file_type}: {path}")
    
    return {
        'regression_results': regression_results,
        'classification_results': classification_results,
        'feature_cols': feature_cols,
        'saved_files': saved_files,
        'ml_framework': ml_framework
    }

def analyze_model_performance(results):
    """Analyze and summarize model performance"""
    print("\n🎯 Analyzing Model Performance and Generating Insights...")
    
    insights = []
    
    # Regression model analysis
    reg_results = results['regression_results']
    if reg_results:
        # Find best regression model
        best_reg_model = max(reg_results.keys(), key=lambda x: reg_results[x]['cv_r2_mean'])
        best_reg_r2 = reg_results[best_reg_model]['cv_r2_mean']
        
        insights.append(f"📈 Best regression model: {best_reg_model} (R² = {best_reg_r2:.4f})")
        
        # Check if any model has good predictive power
        if best_reg_r2 > 0.1:
            insights.append(f"✅ Strong predictive signal detected - {best_reg_model} explains {best_reg_r2*100:.1f}% of variance")
        elif best_reg_r2 > 0.05:
            insights.append(f"⚖️ Moderate predictive signal - {best_reg_model} explains {best_reg_r2*100:.1f}% of variance")
        else:
            insights.append(f"⚠️ Weak predictive signal - market efficiency suggests limited predictability")
        
        # Compare linear vs non-linear models
        linear_models = [name for name in reg_results.keys() if 'Regression' in name]
        nonlinear_models = [name for name in reg_results.keys() if name not in linear_models]
        
        if linear_models and nonlinear_models:
            best_linear_r2 = max([reg_results[name]['cv_r2_mean'] for name in linear_models])
            best_nonlinear_r2 = max([reg_results[name]['cv_r2_mean'] for name in nonlinear_models])
            
            if best_nonlinear_r2 > best_linear_r2 + 0.02:
                insights.append(f"🌲 Non-linear relationships detected - tree models outperform linear by {(best_nonlinear_r2-best_linear_r2)*100:.1f}%")
            else:
                insights.append(f"📏 Market relationships appear mostly linear - minimal non-linear advantage")
    
    # Classification model analysis
    class_results = results['classification_results']
    if class_results:
        # Find best classification model
        best_class_model = max(class_results.keys(), key=lambda x: class_results[x]['cv_f1_mean'])
        best_class_f1 = class_results[best_class_model]['cv_f1_mean']
        best_class_acc = class_results[best_class_model]['cv_accuracy_mean']
        
        insights.append(f"🎯 Best classification model: {best_class_model} (F1 = {best_class_f1:.4f}, Accuracy = {best_class_acc:.4f})")
        
        # Check if better than random
        if best_class_acc > 0.55:
            insights.append(f"📊 Directional prediction shows promise - {best_class_acc*100:.1f}% accuracy beats random (50%)")
        elif best_class_acc > 0.52:
            insights.append(f"⚖️ Slight directional edge detected - {best_class_acc*100:.1f}% accuracy")
        else:
            insights.append(f"🎲 Directional prediction challenging - {best_class_acc*100:.1f}% accuracy near random")
    
    # Feature importance insights
    if reg_results and 'Random_Forest' in reg_results:
        rf_importance = reg_results['Random_Forest'].get('feature_importance')
        if rf_importance is not None:
            top_feature = rf_importance.iloc[0]
            top_category = 'technical' if any(x in top_feature['feature'] for x in ['rsi', 'macd', 'bb_', 'stoch']) else \
                          'momentum' if any(x in top_feature['feature'] for x in ['momentum', 'return']) else \
                          'price' if any(x in top_feature['feature'] for x in ['close', 'sma', 'ema']) else \
                          'volume' if 'volume' in top_feature['feature'] else 'other'
            
            insights.append(f"🔍 Most predictive feature: {top_feature['feature']} ({top_category} indicator)")
            
            # Check feature diversity
            top_5_features = rf_importance.head(5)['feature'].tolist()
            feature_types = set()
            for feat in top_5_features:
                if any(x in feat for x in ['rsi', 'macd', 'bb_', 'stoch']):
                    feature_types.add('technical')
                elif any(x in feat for x in ['momentum', 'return']):
                    feature_types.add('momentum')
                elif any(x in feat for x in ['close', 'sma', 'ema']):
                    feature_types.add('price')
                elif 'volume' in feat:
                    feature_types.add('volume')
            
            if len(feature_types) >= 3:
                insights.append(f"🌐 Diverse signal sources - top features span {len(feature_types)} categories")
            else:
                insights.append(f"🎯 Concentrated signal - top features focus on {', '.join(feature_types)}")
    
    print("🔍 Key Model Performance Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    return insights

def generate_trading_strategy_recommendations(results):
    """Generate practical trading strategy recommendations"""
    print("\n💡 Generating Trading Strategy Recommendations...")
    
    recommendations = []
    
    reg_results = results['regression_results']
    class_results = results['classification_results']
    
    # Strategy recommendations based on model performance
    if reg_results:
        best_reg_model = max(reg_results.keys(), key=lambda x: reg_results[x]['cv_r2_mean'])
        best_reg_r2 = reg_results[best_reg_model]['cv_r2_mean']
        
        if best_reg_r2 > 0.1:
            recommendations.append(f"🎯 Consider quantitative strategy using {best_reg_model} for return prediction")
            recommendations.append(f"📊 Focus on stocks with strong feature signals for higher accuracy")
        elif best_reg_r2 > 0.05:
            recommendations.append(f"⚖️ Use {best_reg_model} for portfolio optimization rather than direct trading")
            recommendations.append(f"🔄 Combine with risk management due to moderate predictive power")
        else:
            recommendations.append(f"🛡️ Focus on risk management and diversification over return prediction")
    
    if class_results:
        best_class_model = max(class_results.keys(), key=lambda x: class_results[x]['cv_f1_mean'])
        best_class_acc = class_results[best_class_model]['cv_accuracy_mean']
        
        if best_class_acc > 0.55:
            recommendations.append(f"🎲 Use {best_class_model} for directional trading signals")
            recommendations.append(f"⚡ Implement momentum strategies based on predicted direction")
        else:
            recommendations.append(f"🔄 Focus on mean reversion rather than momentum strategies")
    
    # Feature-based recommendations
    if reg_results and 'Random_Forest' in reg_results:
        rf_importance = reg_results['Random_Forest'].get('feature_importance')
        if rf_importance is not None:
            top_features = rf_importance.head(5)['feature'].tolist()
            
            if any('volatility' in feat or 'atr' in feat for feat in top_features):
                recommendations.append(f"📊 Incorporate volatility-based position sizing")
            
            if any('momentum' in feat for feat in top_features):
                recommendations.append(f"🚀 Momentum signals show predictive power - consider trend following")
            
            if any('volume' in feat for feat in top_features):
                recommendations.append(f"📈 Volume analysis crucial - integrate into entry/exit rules")
    
    # Risk management recommendations
    recommendations.append(f"🛡️ Implement stop-loss at 2-3% based on volatility analysis")
    recommendations.append(f"📊 Use position sizing based on predicted volatility")
    recommendations.append(f"🔄 Rebalance portfolio monthly based on model predictions")
    
    print("💡 Trading Strategy Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return recommendations

def main():
    """Main execution function for Day 6"""
    
    print("🚀 Stock Market Prediction Engine - Day 6")
    print("Baseline Machine Learning Model Development")
    print("=" * 70)
    
    # Check dependencies from previous days
    config = Config()
    features_path = config.FEATURES_DATA_PATH / "selected_features.csv"
    
    if not features_path.exists():
        print("\n❌ Feature dataset not found!")
        print("Please run Day 4 first to generate engineered features")
        return
    
    # Run baseline ML modeling
    results = run_baseline_ml_modeling()
    
    if results is None:
        print("\n❌ ML model development failed!")
        return
    
    # Analyze model performance
    insights = analyze_model_performance(results)
    
    # Generate trading recommendations
    recommendations = generate_trading_strategy_recommendations(results)
    
    # Display final summary
    reg_results = results['regression_results']
    class_results = results['classification_results']
    
    print("\n🎯 Day 6 Completed Successfully!")
    print("=" * 70)
    print("✅ Baseline machine learning models developed")
    print("✅ Time-series cross-validation implemented")
    print("✅ Model performance evaluation completed")
    print("✅ Feature importance analysis conducted")
    print("✅ Model comparison visualizations created")
    print("✅ Trading strategy recommendations generated")
    
    print(f"\n📊 Final Model Development Summary:")
    print(f"   Regression models trained: {len(reg_results) if reg_results else 0}")
    print(f"   Classification models trained: {len(class_results) if class_results else 0}")
    print(f"   Features used: {len(results['feature_cols'])}")
    print(f"   Cross-validation folds: 5 (time-series aware)")
    
    # Best model summary
    if reg_results:
        best_reg = max(reg_results.keys(), key=lambda x: reg_results[x]['cv_r2_mean'])
        best_reg_score = reg_results[best_reg]['cv_r2_mean']
        print(f"   Best regression model: {best_reg} (R² = {best_reg_score:.4f})")
    
    if class_results:
        best_class = max(class_results.keys(), key=lambda x: class_results[x]['cv_f1_mean'])
        best_class_score = class_results[best_class]['cv_f1_mean']
        print(f"   Best classification model: {best_class} (F1 = {best_class_score:.4f})")
    
    print(f"\n🔍 Key Insights Generated: {len(insights)}")
    print(f"💡 Trading Recommendations: {len(recommendations)}")
    
    print("\n📋 Ready for Day 7:")
    print("1. Advanced ML models (XGBoost, LightGBM, Neural Networks)")
    print("2. Hyperparameter optimization with Optuna")
    print("3. Model ensemble and stacking techniques")
    print("4. Advanced feature selection methods")

if __name__ == "__main__":
    main()