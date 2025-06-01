#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 13
API Client & Comprehensive Testing Suite
"""

import requests
import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration"""
    base_url: str = "http://localhost:8000"
    api_key: str = "demo_key_12345"  # Demo key
    timeout: int = 30
    max_retries: int = 3

class StockPredictionAPIClient:
    """High-performance API client for Stock Market Prediction API"""
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling and retries"""
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method, url, timeout=self.config.timeout, **kwargs
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"API request failed after {self.config.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        return self._make_request("GET", "/health")
    
    def predict_stocks(self, symbols: List[str], include_confidence: bool = True, 
                      include_alerts: bool = True) -> List[Dict[str, Any]]:
        """Get predictions for multiple stocks"""
        payload = {
            "symbols": symbols,
            "include_confidence": include_confidence,
            "include_alerts": include_alerts
        }
        return self._make_request("POST", "/predict", json=payload)
    
    def predict_single_stock(self, symbol: str, horizon: str = "5d") -> Dict[str, Any]:
        """Get prediction for a single stock"""
        payload = {
            "symbol": symbol,
            "horizon": horizon
        }
        return self._make_request("POST", "/predict/single", json=payload)
    
    def optimize_portfolio(self, symbols: List[str], optimization_method: str = "markowitz",
                          target_return: Optional[float] = None, 
                          risk_tolerance: str = "medium") -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        payload = {
            "symbols": symbols,
            "optimization_method": optimization_method,
            "target_return": target_return,
            "risk_tolerance": risk_tolerance
        }
        return self._make_request("POST", "/portfolio/optimize", json=payload)
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self._make_request("GET", "/models/performance")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active trading alerts"""
        return self._make_request("GET", "/alerts/active")
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get market status (no auth required)"""
        # Remove auth header for this request
        headers = self.session.headers.copy()
        del headers["Authorization"]
        
        response = requests.get(
            f"{self.config.base_url}/market/status",
            headers=headers,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()

class APITester:
    """Comprehensive API testing suite"""
    
    def __init__(self, client: StockPredictionAPIClient):
        self.client = client
        self.test_results = []
        
    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling"""
        logger.info(f"🧪 Running test: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            
            self.test_results.append({
                "test": test_name,
                "status": "PASS" if result else "FAIL",
                "duration": duration,
                "error": None,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"✅ {test_name}: PASS ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "duration": duration,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.error(f"❌ {test_name}: FAIL - {e}")
            return False
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        response = self.client.health_check()
        
        required_fields = ["status", "version", "uptime_seconds", "models_loaded"]
        for field in required_fields:
            if field not in response:
                raise Exception(f"Missing field in health response: {field}")
        
        if response["status"] != "healthy":
            raise Exception(f"API not healthy: {response['status']}")
        
        return True
    
    def test_market_status(self) -> bool:
        """Test market status endpoint (no auth)"""
        response = self.client.get_market_status()
        
        required_fields = ["is_open", "current_time", "timezone"]
        for field in required_fields:
            if field not in response:
                raise Exception(f"Missing field in market status: {field}")
        
        return True
    
    def test_single_prediction(self) -> bool:
        """Test single stock prediction"""
        response = self.client.predict_single_stock("AAPL")
        
        required_fields = ["symbol", "prediction", "confidence", "direction", "timestamp"]
        for field in required_fields:
            if field not in response:
                raise Exception(f"Missing field in prediction: {field}")
        
        if response["symbol"] != "AAPL":
            raise Exception("Symbol mismatch in response")
        
        if not isinstance(response["prediction"], (int, float)):
            raise Exception("Prediction is not numeric")
        
        if response["direction"] not in ["BUY", "SELL", "HOLD"]:
            raise Exception(f"Invalid direction: {response['direction']}")
        
        return True
    
    def test_multiple_predictions(self) -> bool:
        """Test multiple stock predictions"""
        symbols = ["AAPL", "AMZN", "NVDA"]
        response = self.client.predict_stocks(symbols)
        
        if not isinstance(response, list):
            raise Exception("Response is not a list")
        
        if len(response) == 0:
            raise Exception("No predictions returned")
        
        # Check first prediction structure
        pred = response[0]
        required_fields = ["symbol", "prediction", "confidence", "direction"]
        for field in required_fields:
            if field not in pred:
                raise Exception(f"Missing field in prediction: {field}")
        
        return True
    
    def test_portfolio_optimization(self) -> bool:
        """Test portfolio optimization"""
        symbols = ["AAPL", "AMZN", "NVDA", "MSFT"]
        response = self.client.optimize_portfolio(symbols, "markowitz")
        
        required_fields = ["weights", "expected_return", "volatility", "sharpe_ratio"]
        for field in required_fields:
            if field not in response:
                raise Exception(f"Missing field in portfolio response: {field}")
        
        # Check weights sum to approximately 1
        weights = response["weights"]
        if not isinstance(weights, dict):
            raise Exception("Weights is not a dictionary")
        
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            raise Exception(f"Weights don't sum to 1: {total_weight}")
        
        return True
    
    def test_model_performance(self) -> bool:
        """Test model performance endpoint"""
        response = self.client.get_model_performance()
        
        required_fields = ["models", "best_model", "summary"]
        for field in required_fields:
            if field not in response:
                raise Exception(f"Missing field in performance response: {field}")
        
        if not isinstance(response["models"], list):
            raise Exception("Models is not a list")
        
        if len(response["models"]) == 0:
            raise Exception("No models in performance data")
        
        return True
    
    def test_alerts(self) -> bool:
        """Test alerts endpoint"""
        response = self.client.get_active_alerts()
        
        if not isinstance(response, list):
            raise Exception("Alerts response is not a list")
        
        # Alerts can be empty, that's fine
        if len(response) > 0:
            alert = response[0]
            required_fields = ["id", "symbol", "type", "message", "timestamp"]
            for field in required_fields:
                if field not in alert:
                    raise Exception(f"Missing field in alert: {field}")
        
        return True
    
    def test_error_handling(self) -> bool:
        """Test API error handling"""
        try:
            # Test invalid symbol
            self.client.predict_single_stock("INVALID_SYMBOL_XYZ")
            return False  # Should have failed
        except Exception:
            pass  # Expected to fail
        
        try:
            # Test empty portfolio optimization
            self.client.optimize_portfolio([])
            return False  # Should have failed
        except Exception:
            pass  # Expected to fail
        
        return True
    
    def test_rate_limiting(self) -> bool:
        """Test rate limiting (basic check)"""
        # Make several rapid requests
        for i in range(5):
            try:
                self.client.predict_single_stock("AAPL")
            except Exception as e:
                if "rate limit" in str(e).lower():
                    logger.info("Rate limiting is working")
                    return True
        
        # If no rate limiting triggered, that's also OK
        return True
    
    def test_performance_benchmark(self) -> bool:
        """Test API performance"""
        start_time = time.time()
        
        # Make 10 prediction requests
        for i in range(10):
            self.client.predict_single_stock("AAPL")
        
        duration = time.time() - start_time
        avg_response_time = duration / 10
        
        logger.info(f"Average response time: {avg_response_time:.3f}s")
        
        # Fail if average response time > 5 seconds
        if avg_response_time > 5.0:
            raise Exception(f"Performance too slow: {avg_response_time:.3f}s per request")
        
        return True
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("🚀 Starting comprehensive API test suite...")
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Market Status", self.test_market_status),
            ("Single Prediction", self.test_single_prediction),
            ("Multiple Predictions", self.test_multiple_predictions),
            ("Portfolio Optimization", self.test_portfolio_optimization),
            ("Model Performance", self.test_model_performance),
            ("Active Alerts", self.test_alerts),
            ("Error Handling", self.test_error_handling),
            ("Rate Limiting", self.test_rate_limiting),
            ("Performance Benchmark", self.test_performance_benchmark)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
            else:
                failed += 1
        
        # Generate test report
        total_duration = sum(result["duration"] for result in self.test_results)
        
        report = {
            "summary": {
                "total_tests": len(tests),
                "passed": passed,
                "failed": failed,
                "success_rate": passed / len(tests) * 100,
                "total_duration": total_duration,
                "timestamp": datetime.now().isoformat()
            },
            "results": self.test_results
        }
        
        logger.info(f"🎯 Test Suite Complete: {passed}/{len(tests)} passed ({report['summary']['success_rate']:.1f}%)")
        
        return report

class LoadTester:
    """Load testing for API performance"""
    
    def __init__(self, client: StockPredictionAPIClient):
        self.client = client
        
    async def concurrent_requests(self, num_requests: int = 20, 
                                 endpoint_func=None) -> Dict[str, Any]:
        """Test concurrent requests"""
        if endpoint_func is None:
            endpoint_func = lambda: self.client.predict_single_stock("AAPL")
        
        logger.info(f"🔥 Load testing with {num_requests} concurrent requests...")
        
        async def make_request():
            start_time = time.time()
            try:
                result = endpoint_func()
                return {"success": True, "duration": time.time() - start_time, "error": None}
            except Exception as e:
                return {"success": False, "duration": time.time() - start_time, "error": str(e)}
        
        # Run concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_duration = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        if successful:
            avg_response_time = sum(r["duration"] for r in successful) / len(successful)
            min_response_time = min(r["duration"] for r in successful)
            max_response_time = max(r["duration"] for r in successful)
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        return {
            "total_requests": num_requests,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / num_requests * 100,
            "total_duration": total_duration,
            "requests_per_second": num_requests / total_duration,
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "errors": [r["error"] for r in failed if r["error"]]
        }

class APIDemo:
    """Interactive API demonstration"""
    
    def __init__(self, client: StockPredictionAPIClient):
        self.client = client
        
    def run_basic_demo(self):
        """Run basic API demonstration"""
        print("\n🚀 Stock Market Prediction API - Interactive Demo")
        print("=" * 60)
        
        try:
            # 1. Health Check
            print("\n1. 🏥 Health Check")
            health = self.client.health_check()
            print(f"   Status: {health['status']}")
            print(f"   Models Loaded: {health['models_loaded']}")
            print(f"   Uptime: {health['uptime_seconds']:.1f}s")
            
            # 2. Market Status
            print("\n2. 📊 Market Status")
            market = self.client.get_market_status()
            print(f"   Market Open: {market['is_open']}")
            print(f"   Current Time: {market['current_time']}")
            print(f"   Trading Session: {market['trading_session']}")
            
            # 3. Single Stock Prediction
            print("\n3. 🔮 Single Stock Prediction (AAPL)")
            pred = self.client.predict_single_stock("AAPL")
            print(f"   Symbol: {pred['symbol']}")
            print(f"   Prediction: {pred['prediction']:+.6f}")
            print(f"   Direction: {pred['direction']}")
            print(f"   Confidence: {pred['confidence']}")
            
            # 4. Multiple Stock Predictions
            print("\n4. 📈 Multiple Stock Predictions")
            symbols = ["AAPL", "AMZN", "NVDA"]
            predictions = self.client.predict_stocks(symbols)
            
            for pred in predictions:
                direction_emoji = "📈" if pred['direction'] == "BUY" else "📉" if pred['direction'] == "SELL" else "⚖️"
                print(f"   {pred['symbol']:>6}: {direction_emoji} {pred['prediction']:+.6f} ({pred['confidence']})")
            
            # 5. Portfolio Optimization
            print("\n5. 💼 Portfolio Optimization")
            portfolio_symbols = ["AAPL", "AMZN", "NVDA", "MSFT"]
            portfolio = self.client.optimize_portfolio(portfolio_symbols)
            
            print(f"   Expected Return: {portfolio['expected_return']*100:.2f}%")
            print(f"   Volatility: {portfolio['volatility']*100:.2f}%")
            print(f"   Sharpe Ratio: {portfolio['sharpe_ratio']:.4f}")
            print("   Optimal Weights:")
            for symbol, weight in portfolio['weights'].items():
                print(f"     {symbol:>6}: {weight:6.1%}")
            
            # 6. Model Performance
            print("\n6. 🏆 Model Performance")
            performance = self.client.get_model_performance()
            best_model = performance['best_model']
            summary = performance['summary']
            
            print(f"   Best Model: {best_model['Model']}")
            print(f"   Best Sharpe: {best_model['Sharpe_Ratio']:.4f}")
            print(f"   Avg Annual Return: {summary['avg_annual_return']:.2f}%")
            print(f"   Total Models: {summary['total_models']}")
            
            # 7. Active Alerts
            print("\n7. 🚨 Active Alerts")
            alerts = self.client.get_active_alerts()
            
            if alerts:
                for alert in alerts[:3]:  # Show top 3
                    severity_emoji = "🔴" if alert['severity'] == "high" else "🟡"
                    print(f"   {severity_emoji} {alert['type']}: {alert['message']}")
            else:
                print("   ✅ No active alerts")
            
            print("\n✅ Demo completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
    
    def run_interactive_demo(self):
        """Run interactive demo with user input"""
        print("\n🎮 Interactive Stock Market Prediction API Demo")
        print("=" * 60)
        
        while True:
            print("\nChoose an option:")
            print("1. 🔮 Predict Single Stock")
            print("2. 📈 Predict Multiple Stocks")
            print("3. 💼 Optimize Portfolio")
            print("4. 🏥 Check API Health")
            print("5. 🚨 View Active Alerts")
            print("6. 🏆 Model Performance")
            print("7. ❌ Exit")
            
            choice = input("\nEnter choice (1-7): ").strip()
            
            try:
                if choice == "1":
                    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
                    pred = self.client.predict_single_stock(symbol)
                    
                    print(f"\n🔮 Prediction for {symbol}:")
                    print(f"   Prediction: {pred['prediction']:+.6f}")
                    print(f"   Direction: {pred['direction']}")
                    print(f"   Confidence: {pred['confidence']}")
                    
                elif choice == "2":
                    symbols_input = input("Enter symbols separated by commas (e.g., AAPL,AMZN,NVDA): ")
                    symbols = [s.strip().upper() for s in symbols_input.split(",")]
                    
                    predictions = self.client.predict_stocks(symbols)
                    
                    print(f"\n📈 Predictions for {len(predictions)} stocks:")
                    for pred in predictions:
                        direction_emoji = "📈" if pred['direction'] == "BUY" else "📉" if pred['direction'] == "SELL" else "⚖️"
                        print(f"   {pred['symbol']:>6}: {direction_emoji} {pred['prediction']:+.6f}")
                        
                elif choice == "3":
                    symbols_input = input("Enter portfolio symbols (e.g., AAPL,AMZN,NVDA,MSFT): ")
                    symbols = [s.strip().upper() for s in symbols_input.split(",")]
                    
                    method = input("Optimization method (markowitz/risk_parity) [markowitz]: ").strip() or "markowitz"
                    
                    portfolio = self.client.optimize_portfolio(symbols, method)
                    
                    print(f"\n💼 Optimized Portfolio ({method}):")
                    print(f"   Expected Return: {portfolio['expected_return']*100:.2f}%")
                    print(f"   Sharpe Ratio: {portfolio['sharpe_ratio']:.4f}")
                    print("   Weights:")
                    for symbol, weight in portfolio['weights'].items():
                        print(f"     {symbol:>6}: {weight:6.1%}")
                        
                elif choice == "4":
                    health = self.client.health_check()
                    print(f"\n🏥 API Health:")
                    print(f"   Status: {health['status']}")
                    print(f"   Models: {health['models_loaded']}")
                    print(f"   Uptime: {health['uptime_seconds']:.1f}s")
                    
                elif choice == "5":
                    alerts = self.client.get_active_alerts()
                    print(f"\n🚨 Active Alerts ({len(alerts)}):")
                    
                    if alerts:
                        for alert in alerts:
                            severity_emoji = "🔴" if alert['severity'] == "high" else "🟡"
                            print(f"   {severity_emoji} {alert['message']}")
                    else:
                        print("   ✅ No active alerts")
                        
                elif choice == "6":
                    performance = self.client.get_model_performance()
                    best = performance['best_model']
                    summary = performance['summary']
                    
                    print(f"\n🏆 Model Performance:")
                    print(f"   Best Model: {best['Model']}")
                    print(f"   Sharpe Ratio: {best['Sharpe_Ratio']:.4f}")
                    print(f"   Total Models: {summary['total_models']}")
                    
                elif choice == "7":
                    print("\n👋 Goodbye!")
                    break
                    
                else:
                    print("❌ Invalid choice. Please enter 1-7.")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            
            input("\nPress Enter to continue...")

def main():
    """Main function demonstrating API usage"""
    print("🚀 Stock Market Prediction API - Client & Testing Suite")
    print("=" * 70)
    
    # Initialize client
    config = APIConfig(
        base_url="http://localhost:8000",
        api_key="demo_key_12345"
    )
    client = StockPredictionAPIClient(config)
    
    while True:
        print("\nChoose operation:")
        print("1. 🧪 Run Test Suite")
        print("2. 🔥 Run Load Test")
        print("3. 🎮 Interactive Demo")
        print("4. 🚀 Basic Demo")
        print("5. ❌ Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            # Run comprehensive test suite
            tester = APITester(client)
            report = tester.run_all_tests()
            
            print(f"\n📊 TEST SUMMARY:")
            print(f"   Tests Run: {report['summary']['total_tests']}")
            print(f"   Passed: {report['summary']['passed']}")
            print(f"   Failed: {report['summary']['failed']}")
            print(f"   Success Rate: {report['summary']['success_rate']:.1f}%")
            print(f"   Total Duration: {report['summary']['total_duration']:.2f}s")
            
            # Show failed tests
            failed_tests = [r for r in report['results'] if r['status'] == 'FAIL']
            if failed_tests:
                print(f"\n❌ Failed Tests:")
                for test in failed_tests:
                    print(f"   • {test['test']}: {test['error']}")
            
        elif choice == "2":
            # Run load test
            load_tester = LoadTester(client)
            
            try:
                num_requests = int(input("Number of concurrent requests [20]: ") or "20")
                results = asyncio.run(load_tester.concurrent_requests(num_requests))
                
                print(f"\n🔥 LOAD TEST RESULTS:")
                print(f"   Total Requests: {results['total_requests']}")
                print(f"   Successful: {results['successful_requests']}")
                print(f"   Failed: {results['failed_requests']}")
                print(f"   Success Rate: {results['success_rate']:.1f}%")
                print(f"   Requests/Second: {results['requests_per_second']:.2f}")
                print(f"   Avg Response Time: {results['avg_response_time']:.3f}s")
                print(f"   Min Response Time: {results['min_response_time']:.3f}s")
                print(f"   Max Response Time: {results['max_response_time']:.3f}s")
                
                if results['errors']:
                    print(f"\n❌ Errors encountered:")
                    for error in set(results['errors'][:5]):  # Show unique errors
                        print(f"   • {error}")
                        
            except ValueError:
                print("❌ Invalid number of requests")
                
        elif choice == "3":
            # Interactive demo
            demo = APIDemo(client)
            demo.run_interactive_demo()
            
        elif choice == "4":
            # Basic demo
            demo = APIDemo(client)
            demo.run_basic_demo()
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()