#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 9
Ensemble Methods & Model Stacking
"""

from src.config import Config
from src.ensemble_models import EnsembleFramework
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main execution function for Day 9"""
    
    print("🚀 Stock Market Prediction Engine - Day 9")
    print("Ensemble Methods & Model Stacking")
    print("=" * 50)
    
    # Initialize ensemble framework
    ensemble_framework = EnsembleFramework()
    
    # Phase 1: Load Base Models and Data
    print("\n📊 Phase 1: Loading Base Models and Data")
    print("-" * 40)
    models, df, X, y = ensemble_framework.load_base_models()
    print(f"✅ Loaded {len(models)} base models")
    print(f"📈 Dataset: {len(df):,} records with {X.shape[1]} features")
    
    # Phase 2: Evaluate Individual Models
    print("\n📈 Phase 2: Individual Model Evaluation")
    print("-" * 40)
    individual_results = ensemble_framework.evaluate_individual_models(X, y)
    print(f"✅ Evaluated {len(individual_results)} individual models")
    
    # Phase 3: Create Ensemble Models
    print("\n🤖 Phase 3: Creating Ensemble Models")
    print("-" * 40)
    ensembles = ensemble_framework.create_all_ensembles(X, y)
    print(f"✅ Created {len(ensembles)} ensemble models:")
    for name in ensembles.keys():
        print(f"   • {name}")
    
    # Phase 4: Evaluate Ensemble Models
    print("\n📊 Phase 4: Ensemble Model Evaluation")
    print("-" * 40)
    ensemble_results = ensemble_framework.evaluate_all_ensembles(X, y)
    print(f"✅ Evaluated {len(ensemble_results)} ensemble models")
    
    # Phase 5: Create Visualizations
    print("\n🎨 Phase 5: Creating Visualizations")
    print("-" * 40)
    fig = ensemble_framework.create_visualizations()
    print("✅ Interactive visualizations created")
    
    # Phase 6: Save Models and Results
    print("\n💾 Phase 6: Saving Models and Results")
    print("-" * 40)
    saved_models = ensemble_framework.save_ensemble_models()
    results_path, report_path = ensemble_framework.save_comprehensive_results(saved_models)
    print(f"✅ Saved {len(saved_models)} ensemble models")
    print("✅ Comprehensive results and reports saved")
    
    # Final Performance Summary
    print("\n🏆 Final Performance Summary")
    print("=" * 50)
    
    summary = ensemble_framework.get_performance_summary()
    
    if summary:
        print(f"🥇 Best Individual Model: {summary['best_individual']['name']}")
        print(f"   R² Score: {summary['best_individual']['score']:.4f}")
        print(f"   RMSE: {summary['best_individual']['rmse']:.4f}")
        
        print(f"\n🏆 Best Ensemble Model: {summary['best_ensemble']['name']}")
        print(f"   R² Score: {summary['best_ensemble']['score']:.4f}")
        print(f"   RMSE: {summary['best_ensemble']['rmse']:.4f}")
        
        print(f"\n📈 Ensemble Improvement: {summary['improvement_pct']:+.1f}%")
        
        if summary['improvement_pct'] > 10:
            print("   🎉 Significant improvement achieved!")
        elif summary['improvement_pct'] > 0:
            print("   👍 Moderate improvement achieved")
        else:
            print("   ⚠️ Limited improvement over baseline")
    
    # Display saved files
    print(f"\n📁 Files Created:")
    print(f"   📊 Visualizations: plots/day9_ensemble_analysis.html")
    print(f"   📈 Results: {results_path.name}")
    print(f"   📄 Report: {report_path.name}")
    print(f"   🤖 Ensemble Models: {len(saved_models)} models saved")
    
    # Show sklearn libraries used
    print(f"\n🔧 Sklearn Libraries Used:")
    print("   ✅ sklearn.ensemble.VotingRegressor")
    print("   ✅ sklearn.linear_model.LinearRegression")
    print("   ✅ sklearn.neural_network.MLPRegressor")
    print("   ✅ sklearn.model_selection.cross_val_score")
    print("   ✅ sklearn.model_selection.TimeSeriesSplit")
    print("   ✅ sklearn.preprocessing.StandardScaler")
    print("   ✅ sklearn.metrics (r2_score, RMSE, MAE)")
    
    # Success message
    print("\n🎯 Day 9 Completed Successfully!")
    print("=" * 50)
    print("✅ Modular ensemble framework implemented")
    print("✅ All sklearn libraries properly utilized")
    print("✅ Comprehensive visualizations created")
    print("✅ Models saved for production use")
    print("✅ Detailed performance analysis completed")
    
    print("\n📋 Ready for Day 10:")
    print("1. Advanced validation and backtesting")
    print("2. Risk-adjusted performance metrics")
    print("3. Model robustness testing")
    print("4. Production pipeline preparation")

if __name__ == "__main__":
    main()