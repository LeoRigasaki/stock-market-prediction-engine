# 📈 Stock Market Prediction Engine - Technical Concepts & Mathematical Formulas

> **Comprehensive reference of mathematical concepts, formulas, and algorithms implemented in the system**

## 🎯 Overview

This document provides detailed mathematical explanations of all concepts, formulas, and algorithms used throughout the Stock Market Prediction Engine. Each section corresponds to specific source files and their implementations.

---

## 📊 `src/feature_engineer.py` - Feature Engineering Mathematics

### **Basic Price Features**

#### **Price Change & Returns**
$$\text{Price Change} = P_{\text{close}} - P_{\text{open}}$$

$$\text{Price Change \%} = \frac{P_{\text{close}} - P_{\text{open}}}{P_{\text{open}}} \times 100$$

$$\text{Daily Return} = \frac{P_t - P_{t-1}}{P_{t-1}} \times 100$$

#### **Gap Analysis**
$$\text{Gap} = P_{\text{open},t} - P_{\text{close},t-1}$$

$$\text{Gap \%} = \frac{P_{\text{open},t} - P_{\text{close},t-1}}{P_{\text{close},t-1}} \times 100$$

### **Technical Indicators**

#### **Simple Moving Average (SMA)**
$$\text{SMA}_n = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}$$

Where:
- $n$ = period length (5, 10, 20, 50, 100, 200 days)
- $P_t$ = price at time $t$

#### **Exponential Moving Average (EMA)**
$$\text{EMA}_t = \alpha \cdot P_t + (1-\alpha) \cdot \text{EMA}_{t-1}$$

$$\alpha = \frac{2}{n+1}$$

Where:
- $\alpha$ = smoothing factor
- $n$ = period length

#### **Relative Strength Index (RSI)**
$$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$

$$\text{RS} = \frac{\text{Average Gain over n periods}}{\text{Average Loss over n periods}}$$

$$\text{Average Gain} = \frac{1}{n} \sum_{i=1}^{n} \max(P_i - P_{i-1}, 0)$$

$$\text{Average Loss} = \frac{1}{n} \sum_{i=1}^{n} \max(P_{i-1} - P_i, 0)$$

#### **MACD (Moving Average Convergence Divergence)**
$$\text{MACD Line} = \text{EMA}_{12} - \text{EMA}_{26}$$

$$\text{Signal Line} = \text{EMA}_9(\text{MACD Line})$$

$$\text{MACD Histogram} = \text{MACD Line} - \text{Signal Line}$$

#### **Bollinger Bands**
$$\text{Middle Band} = \text{SMA}_{20}$$

$$\text{Upper Band} = \text{SMA}_{20} + (2 \times \sigma_{20})$$

$$\text{Lower Band} = \text{SMA}_{20} - (2 \times \sigma_{20})$$

$$\text{Bollinger Band Width} = \frac{\text{Upper Band} - \text{Lower Band}}{\text{Middle Band}} \times 100$$

$$\text{Bollinger Band Position} = \frac{P_{\text{close}} - \text{Lower Band}}{\text{Upper Band} - \text{Lower Band}}$$

Where $\sigma_{20}$ is the 20-period standard deviation.

#### **Average True Range (ATR)**
$$\text{True Range} = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$$

$$\text{ATR}_n = \frac{1}{n} \sum_{i=0}^{n-1} \text{TR}_{t-i}$$

$$\text{ATR Ratio} = \frac{\text{ATR}}{P_{\text{close}}} \times 100$$

### **Volume Indicators**

#### **Volume Weighted Average Price (VWAP)**
$$\text{VWAP} = \frac{\sum_{i=1}^{n} (P_i \times V_i)}{\sum_{i=1}^{n} V_i}$$

Where:
- $P_i$ = price at period $i$
- $V_i$ = volume at period $i$

#### **On Balance Volume (OBV)**
$$\text{OBV}_t = \text{OBV}_{t-1} + \begin{cases} 
V_t & \text{if } C_t > C_{t-1} \\
-V_t & \text{if } C_t < C_{t-1} \\
0 & \text{if } C_t = C_{t-1}
\end{cases}$$

### **Volatility Measures**
$$\text{Historical Volatility}_{20d} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (R_i - \bar{R})^2} \times \sqrt{252}$$

Where:
- $R_i$ = daily return at day $i$
- $\bar{R}$ = mean daily return
- $252$ = annualization factor

---

## 🤖 `src/ml_models.py` & `src/advanced_models.py` - Machine Learning Mathematics

### **Model Evaluation Metrics**

#### **Regression Metrics**
$$\text{Mean Squared Error (MSE)} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

$$\text{Root Mean Squared Error (RMSE)} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

$$\text{Mean Absolute Error (MAE)} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

$$\text{R-squared (R²)} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

#### **Classification Metrics**
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### **XGBoost Mathematics**

#### **Gradient Boosting Objective**
$$\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

Where:
- $l$ = loss function
- $\Omega$ = regularization term
- $f_k$ = $k$-th tree

#### **Taylor Expansion for Optimization**
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$

Where:
- $g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$ (first-order gradient)
- $h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$ (second-order gradient)

---

## 🏛️ `src/ensemble_models.py` - Ensemble Mathematics

### **Voting Ensemble**
$$\hat{y}_{\text{ensemble}} = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_m$$

For weighted voting:
$$\hat{y}_{\text{ensemble}} = \frac{\sum_{m=1}^{M} w_m \hat{y}_m}{\sum_{m=1}^{M} w_m}$$

Where:
- $M$ = number of models
- $w_m$ = weight for model $m$
- $\hat{y}_m$ = prediction from model $m$

### **Stacking Ensemble**
$$\hat{y}_{\text{stacking}} = g(f_1(x), f_2(x), ..., f_M(x))$$

Where:
- $g$ = meta-learner function
- $f_m(x)$ = base model $m$ prediction

### **Out-of-Fold Predictions**
For cross-validation with $K$ folds:
$$\text{OOF}_i = f^{(-k(i))}(x_i)$$

Where $f^{(-k(i))}$ is the model trained on all folds except $k(i)$ (the fold containing sample $i$).

---

## 💼 `src/risk_management.py` - Financial Risk Mathematics

### **Value at Risk (VaR)**

#### **Historical VaR**
$$\text{VaR}_\alpha = \text{Percentile}(R, \alpha \times 100\%)$$

Where:
- $R$ = return distribution
- $\alpha$ = confidence level (e.g., 0.05 for 95% VaR)

#### **Parametric VaR (Normal Distribution)**
$$\text{VaR}_\alpha = \mu + \sigma \times \Phi^{-1}(\alpha)$$

Where:
- $\mu$ = mean return
- $\sigma$ = standard deviation of returns
- $\Phi^{-1}$ = inverse cumulative standard normal distribution

### **Conditional Value at Risk (CVaR)**
$$\text{CVaR}_\alpha = E[R | R \leq \text{VaR}_\alpha]$$

### **Maximum Drawdown**
$$\text{Drawdown}_t = \frac{\text{Peak}_t - \text{Value}_t}{\text{Peak}_t}$$

$$\text{Maximum Drawdown} = \max_{t} \text{Drawdown}_t$$

Where:
$$\text{Peak}_t = \max_{s \leq t} \text{Value}_s$$

### **Risk-Adjusted Performance Ratios**

#### **Sharpe Ratio**
$$\text{Sharpe Ratio} = \frac{E[R] - R_f}{\sigma_R}$$

Where:
- $E[R]$ = expected portfolio return
- $R_f$ = risk-free rate
- $\sigma_R$ = standard deviation of portfolio returns

#### **Sortino Ratio**
$$\text{Sortino Ratio} = \frac{E[R] - R_f}{\sigma_{\text{downside}}}$$

$$\sigma_{\text{downside}} = \sqrt{E[\min(R - R_f, 0)^2]}$$

#### **Calmar Ratio**
$$\text{Calmar Ratio} = \frac{\text{Annual Return}}{\text{Maximum Drawdown}}$$

#### **Information Ratio**
$$\text{Information Ratio} = \frac{E[R - R_b]}{\sigma_{R - R_b}}$$

Where:
- $R_b$ = benchmark return
- $\sigma_{R - R_b}$ = tracking error

### **Portfolio Optimization**

#### **Markowitz Mean-Variance Optimization**
**Objective Function:**
$$\min_w \frac{1}{2} w^T \Sigma w$$

**Subject to:**
$$w^T \mu = \mu_p \quad \text{(target return constraint)}$$
$$w^T \mathbf{1} = 1 \quad \text{(weight sum constraint)}$$
$$w_i \geq 0 \quad \forall i \quad \text{(long-only constraint)}$$

Where:
- $w$ = portfolio weights vector
- $\Sigma$ = covariance matrix
- $\mu$ = expected returns vector
- $\mu_p$ = target portfolio return

#### **Risk Parity Portfolio**
$$w_i = \frac{1/\sigma_i}{\sum_{j=1}^n 1/\sigma_j}$$

Where $\sigma_i$ is the volatility of asset $i$.

### **Position Sizing - Kelly Criterion**
$$f^* = \frac{bp - q}{b}$$

Where:
- $f^*$ = optimal fraction of capital to wager
- $b$ = odds received (average win / average loss)
- $p$ = probability of winning
- $q$ = probability of losing $(1-p)$

---

## 📊 `src/market_analyzer.py` - Statistical Analysis Mathematics

### **Correlation Analysis**
$$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}$$

### **Principal Component Analysis (PCA)**
**Covariance Matrix:**
$$C = \frac{1}{n-1} X^T X$$

**Eigenvalue Decomposition:**
$$C = V \Lambda V^T$$

**Principal Components:**
$$Y = XV$$

**Explained Variance Ratio:**
$$\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$$

### **Statistical Tests**

#### **Jarque-Bera Test for Normality**
$$JB = \frac{n}{6} \left( S^2 + \frac{(K-3)^2}{4} \right)$$

Where:
- $S$ = sample skewness
- $K$ = sample kurtosis
- $n$ = sample size

$$S = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^3}{\left(\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2\right)^{3/2}}$$

$$K = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^4}{\left(\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2\right)^2}$$

### **Anomaly Detection**

#### **Z-Score Method**
$$z_i = \frac{x_i - \mu}{\sigma}$$

Outliers defined as $|z_i| > 3$.

#### **Interquartile Range (IQR) Method**
$$\text{IQR} = Q_3 - Q_1$$

Outliers defined as:
$$x_i < Q_1 - 1.5 \times \text{IQR} \quad \text{or} \quad x_i > Q_3 + 1.5 \times \text{IQR}$$

---

## ⏱️ `src/realtime_prediction.py` - Real-time Processing Mathematics

### **Feature Scaling and Normalization**
$$z = \frac{x - \mu}{\sigma}$$

### **Prediction Confidence Intervals**
For regression with prediction $\hat{y}$ and standard error $SE$:
$$\text{CI} = \hat{y} \pm t_{\alpha/2,df} \times SE$$

### **Exponential Moving Average for Real-time Updates**
$$\text{EMA}_t = \alpha \times x_t + (1-\alpha) \times \text{EMA}_{t-1}$$

With adaptive smoothing factor:
$$\alpha = \frac{2}{N+1}$$

---

## 🔄 `src/validation_framework.py` - Validation Mathematics

### **Walk-Forward Validation**
For time series with $T$ observations:
- Training window: $[t-w, t-1]$
- Test window: $[t, t+h-1]$
- Step size: $s$

### **Statistical Significance Testing**

#### **Paired t-test**
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

Where:
- $\bar{d}$ = mean difference
- $s_d$ = standard deviation of differences
- $n$ = sample size

#### **Wilcoxon Signed-Rank Test**
$$W = \sum_{i=1}^{n} \text{sgn}(x_i) \cdot R_i$$

Where $R_i$ is the rank of $|x_i|$.

#### **Cohen's d (Effect Size)**
$$d = \frac{\mu_1 - \mu_2}{\sigma_{\text{pooled}}}$$

$$\sigma_{\text{pooled}} = \sqrt{\frac{(n_1-1)\sigma_1^2 + (n_2-1)\sigma_2^2}{n_1+n_2-2}}$$

---

## 📈 Summary of Mathematical Concepts

### **Feature Engineering (73 features)**
- 24 Basic price/volume features
- 28 Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- 10 Time-based features with cyclical encoding
- 27 Lag and rolling statistics features
- 6 Target variables for different horizons

### **Machine Learning Models**
- Ensemble of 10+ algorithms
- Hyperparameter optimization with Bayesian methods
- Time-series cross-validation
- Performance evaluation with financial metrics

### **Risk Management**
- VaR and CVaR calculations
- Portfolio optimization (Markowitz, Risk Parity)
- Kelly Criterion for position sizing
- Comprehensive risk-adjusted performance metrics

### **Statistical Validation**
- Walk-forward validation for time series
- Statistical significance testing
- Robustness analysis across market regimes
- Performance attribution analysis

---