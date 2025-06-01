#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 15
System Integration & Testing
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.config import Config
from src.integration_testing import SystemIntegrationTester, SystemOptimizer

def display_banner():
    """Display Day 15 banner"""
    print("Stock Market Prediction Engine - Day 15")
    print("System Integration & Testing")
    print("=" * 50)

def display_test_summary(report: dict):
    """Display test results summary"""
    summary = report['test_summary']
    
    print(f"\nTest Execution Summary:")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Warnings: {summary['warnings']} ⚠️")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Execution Time: {summary['total_execution_time']:.2f}s")
    print(f"System Status: {report['system_status']}")

def display_detailed_results(report: dict):
    """Display detailed test results"""
    print("\nDetailed Test Results:")
    print("-" * 30)
    
    for test_name, result in report['test_results'].items():
        status_icon = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}")
        if result['details']:
            print(f"   {result['details']}")
        if result['execution_time'] > 0:
            print(f"   Time: {result['execution_time']:.2f}s")

def display_performance_metrics(report: dict):
    """Display performance metrics"""
    if report['performance_metrics']:
        print("\nPerformance Metrics:")
        print("-" * 20)
        for metric, value in report['performance_metrics'].items():
            if isinstance(value, float):
                print(f"{metric}: {value:.2f}s")
            else:
                print(f"{metric}: {value}")

def display_recommendations(report: dict):
    """Display system recommendations"""
    if report['recommendations']:
        print("\nSystem Recommendations:")
        print("-" * 25)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

def save_test_report(report: dict) -> str:
    """Save test report to file"""
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = config.PROCESSED_DATA_PATH / f"integration_test_report_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return str(report_path)

async def run_integration_tests():
    """Run comprehensive integration tests"""
    print("Initializing System Integration Tester...")
    
    tester = SystemIntegrationTester()
    
    print("Running comprehensive integration tests...")
    report = await tester.run_comprehensive_tests()
    
    # Display results
    display_test_summary(report)
    display_detailed_results(report)
    display_performance_metrics(report)
    display_recommendations(report)
    
    # Save report
    report_path = save_test_report(report)
    print(f"\nTest report saved: {report_path}")
    
    return report

def run_system_optimization():
    """Run system optimization"""
    print("\nInitializing System Optimizer...")
    
    optimizer = SystemOptimizer()
    
    # Optimize prediction pipeline
    print("Optimizing prediction pipeline...")
    pipeline_results = optimizer.optimize_prediction_pipeline()
    
    if pipeline_results.get('optimizations_applied'):
        print("Pipeline Optimizations:")
        for opt in pipeline_results['optimizations_applied']:
            print(f"✅ {opt}")
    
    if pipeline_results.get('performance_improvements'):
        print("Performance Improvements:")
        for improvement, value in pipeline_results['performance_improvements'].items():
            print(f"📈 {improvement}: {value}")
    
    # Clean up system files
    print("\nCleaning up system files...")
    cleanup_results = optimizer.cleanup_system_files()
    
    if cleanup_results.get('files_cleaned', 0) > 0:
        print(f"✅ Cleaned {cleanup_results['files_cleaned']} files")
        print(f"💾 Freed {cleanup_results['space_freed_mb']:.2f} MB")
    
    return {
        'pipeline_optimization': pipeline_results,
        'cleanup_results': cleanup_results
    }

def validate_system_health():
    """Quick system health validation"""
    print("\nValidating system health...")
    
    config = Config()
    health_checks = {
        'configuration': False,
        'data_files': False,
        'model_files': False,
        'directory_structure': False
    }
    
    # Check configuration
    try:
        config.create_directories()
        health_checks['configuration'] = True
        print("✅ Configuration system")
    except:
        print("❌ Configuration system")
    
    # Check essential data files
    essential_files = [
        config.FEATURES_DATA_PATH / "selected_features.csv",
        config.PROCESSED_DATA_PATH / "target_stocks.txt"
    ]
    
    files_exist = sum(1 for f in essential_files if f.exists())
    if files_exist >= len(essential_files) * 0.5:  # At least 50% exist
        health_checks['data_files'] = True
        print(f"✅ Data files ({files_exist}/{len(essential_files)})")
    else:
        print(f"⚠️ Data files ({files_exist}/{len(essential_files)})")
    
    # Check model files
    models_dir = config.PROJECT_ROOT / "models"
    if models_dir.exists():
        model_files = list(models_dir.rglob("*.joblib"))
        if len(model_files) >= 3:
            health_checks['model_files'] = True
            print(f"✅ Model files ({len(model_files)} found)")
        else:
            print(f"⚠️ Model files ({len(model_files)} found)")
    else:
        print("❌ Model directory missing")
    
    # Check directory structure
    required_dirs = [config.DATA_PATH, config.LOGS_PATH, config.PROJECT_ROOT / "src"]
    dirs_exist = sum(1 for d in required_dirs if d.exists())
    if dirs_exist == len(required_dirs):
        health_checks['directory_structure'] = True
        print("✅ Directory structure")
    else:
        print(f"⚠️ Directory structure ({dirs_exist}/{len(required_dirs)})")
    
    health_score = sum(health_checks.values()) / len(health_checks) * 100
    status = "HEALTHY" if health_score >= 75 else "DEGRADED" if health_score >= 50 else "CRITICAL"
    
    print(f"\nSystem Health Score: {health_score:.0f}% ({status})")
    return health_checks, health_score

def create_deployment_checklist():
    """Create deployment readiness checklist"""
    print("\nGenerating deployment checklist...")
    
    config = Config()
    checklist = {
        'data_pipeline': {
            'feature_data': (config.FEATURES_DATA_PATH / "selected_features.csv").exists(),
            'target_stocks': (config.PROCESSED_DATA_PATH / "target_stocks.txt").exists(),
            'validation_results': (config.PROCESSED_DATA_PATH / "day10_validation_results.json").exists()
        },
        'models': {
            'ensemble_models': (config.PROJECT_ROOT / "models" / "ensemble").exists(),
            'baseline_models': (config.PROJECT_ROOT / "models").exists(),
            'model_artifacts': len(list((config.PROJECT_ROOT / "models").rglob("*.joblib"))) >= 3
        },
        'applications': {
            'dashboard': (config.PROJECT_ROOT / "src" / "streamlit_dashboard.py").exists(),
            'api_server': (config.PROJECT_ROOT / "src" / "api_server.py").exists(),
            'realtime_engine': (config.PROJECT_ROOT / "src" / "realtime_prediction.py").exists()
        },
        'infrastructure': {
            'configuration': (config.PROJECT_ROOT / "src" / "config.py").exists(),
            'logging': config.LOGS_PATH.exists(),
            'documentation': (config.PROJECT_ROOT / "README.md").exists()
        }
    }
    
    print("\nDeployment Readiness Checklist:")
    print("=" * 35)
    
    total_checks = 0
    passed_checks = 0
    
    for category, checks in checklist.items():
        print(f"\n{category.title()}:")
        for check_name, status in checks.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {check_name.replace('_', ' ').title()}")
            total_checks += 1
            if status:
                passed_checks += 1
    
    readiness_score = (passed_checks / total_checks) * 100
    deployment_status = "READY" if readiness_score >= 80 else "NEEDS_WORK" if readiness_score >= 60 else "NOT_READY"
    
    print(f"\nDeployment Readiness: {readiness_score:.0f}% ({deployment_status})")
    
    return checklist, readiness_score

def generate_final_report(test_report: dict, optimization_results: dict, health_checks: dict, deployment_checklist: dict):
    """Generate comprehensive final report"""
    config = Config()
    
    final_report = {
        'day_15_summary': {
            'completion_date': datetime.now().isoformat(),
            'test_success_rate': test_report['test_summary']['success_rate'],
            'system_status': test_report['system_status'],
            'health_score': sum(health_checks.values()) / len(health_checks) * 100,
            'deployment_readiness': sum(1 for category in deployment_checklist.values() 
                                      for status in category.values() if status) / 
                                   sum(len(category) for category in deployment_checklist.values()) * 100
        },
        'integration_testing': test_report,
        'system_optimization': optimization_results,
        'health_validation': health_checks,
        'deployment_checklist': deployment_checklist,
        'next_steps': [
            "Deploy to production environment",
            "Set up monitoring and alerting",
            "Configure automated backups",
            "Implement continuous integration",
            "Schedule model retraining"
        ]
    }
    
    # Save final report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = config.PROCESSED_DATA_PATH / f"day15_final_report_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    return final_report, str(report_path)

async def main():
    """Main execution function for Day 15"""
    
    display_banner()
    
    try:
        # Phase 1: System Health Check
        print("\nPhase 1: System Health Validation")
        print("-" * 40)
        health_checks, health_score = validate_system_health()
        
        # Phase 2: Integration Testing
        print("\nPhase 2: Comprehensive Integration Testing")
        print("-" * 45)
        test_report = await run_integration_tests()
        
        # Phase 3: System Optimization
        print("\nPhase 3: System Optimization")
        print("-" * 35)
        optimization_results = run_system_optimization()
        
        # Phase 4: Deployment Readiness
        print("\nPhase 4: Deployment Readiness Assessment")
        print("-" * 45)
        deployment_checklist, readiness_score = create_deployment_checklist()
        
        # Phase 5: Final Report
        print("\nPhase 5: Final Report Generation")
        print("-" * 35)
        final_report, report_path = generate_final_report(
            test_report, optimization_results, health_checks, deployment_checklist
        )
        
        # Display final summary
        print("\nDay 15 Integration & Testing Complete!")
        print("=" * 45)
        
        summary = final_report['day_15_summary']
        print(f"Test Success Rate: {summary['test_success_rate']:.1f}%")
        print(f"System Health: {summary['health_score']:.0f}%")
        print(f"Deployment Readiness: {summary['deployment_readiness']:.0f}%")
        print(f"Overall Status: {summary['system_status']}")
        
        print(f"\nFinal Report: {report_path}")
        
        # Recommendations
        if test_report.get('recommendations'):
            print("\nCritical Recommendations:")
            for i, rec in enumerate(test_report['recommendations'][:3], 1):
                print(f"{i}. {rec}")
        
        # Next steps
        print(f"\nNext Steps for Day 16:")
        for step in final_report['next_steps'][:3]:
            print(f"• {step}")
        
        print("\nSystem ready for Day 16: Final Deployment & Documentation")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Day 15 execution error: {e}")

if __name__ == "__main__":
    asyncio.run(main())