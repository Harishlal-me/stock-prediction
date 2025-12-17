📈 LSTM Stock Prediction
Professional Multi-Task Trading Signals for Tech Stocks
Show Image
Show Image
Show Image
A production-grade LSTM model delivering calibrated trading signals for 6 major tech stocks with 59.7% next-day and 67.4% weekly directional accuracy, trained on 15+ years of market data.

🎯 Live Trading Signals (Dec 17, 2025)
StockP(Week ↑)SignalActionEdgeAAPL35.9%🔴 DOWN (MEDIUM)SELL14.1%MSFT54.2%⚪ HOLD (LOW)HOLD4.2%NVDA37.2%🔴 DOWN (MEDIUM)SELL12.8%AMZN38.8%🔴 DOWN (MEDIUM)SELL11.2%GOOGL34.6%🔴 DOWN (HIGH)SELL15.4%META30.9%🔴 DOWN (HIGH)SELL19.1%
Market Regime: 🔴 STRONGLY BEARISH (5/6 SELL signals)

🚀 Quick Start
1. Installation
bash# Clone repository
git clone https://github.com/Harishlal-me/stock-prediction.git
cd stock-prediction

# Install dependencies
pip install -r requirements.txt
2. API Configuration
bash# Get FREE API key from EODHD
# Visit: https://eodhd.com → Dashboard → Copy API key

# Configure in src/data_loader.py:
EODHD_API_KEY = "your_api_key_here"
3. Train & Predict
bash# Train model on 15+ years of historical data
python train.py

# Generate trading signals
python predict.py --stock AAPL
Example Output
📈 AAPL PROFESSIONAL TRADING SIGNAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week Prediction: DOWN (MEDIUM CONFIDENCE)
Probability UP: 35.9%

🎯 RECOMMENDED ACTION: SELL
📊 EDGE: 14.1% deviation from neutral (50%)

Model Performance: 67.4% weekly accuracy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ Technical Architecture
Input Features (20+ Technical Indicators)
60-day sequence window × Multi-dimensional features:

Price Action:
├── OHLCV (Open, High, Low, Close, Volume)
├── Daily/Weekly Returns
└── Volume Ratio

Momentum Indicators:
├── RSI(14) - Relative Strength Index
├── MACD - Moving Average Convergence Divergence
└── MACD Signal Line

Trend Indicators:
├── SMA(10, 20, 50) - Simple Moving Averages
├── EMA(12, 26) - Exponential Moving Averages
└── Price-to-MA Ratios

Volatility:
└── Historical Volatility (20-day rolling)

Training Data: 15+ years (2010-present)
LSTM Model Architecture
pythonInput: (batch_size, 60, 20) # 60 days × 20 features

LSTM Architecture:
├── LSTM Layer 1: 64 units (return_sequences=True)
│   └── Captures short-term patterns
├── LSTM Layer 2: 32 units  
│   └── Learns long-term dependencies
├── Dense Layer 1: 64 units (ReLU)
├── Dense Layer 2: 32 units (ReLU)
└── Output Layer: 4 units

Multi-Task Outputs:
├── Tomorrow Direction (binary classification)
├── Week Direction (binary classification) ← PRIMARY SIGNAL
├── Tomorrow Return (regression)
└── Week Return (regression)

📊 Professional Signal Calibration
Probability Thresholds
pythonSignal Generation:
├── BUY:  P(Week ↑) ≥ 55.0%
├── HOLD: 45.0% < P(Week ↑) < 55.0%
└── SELL: P(Week ↑) ≤ 45.0%

Signal Strength (Edge from 50%):
├── HIGH:   |P - 50%| ≥ 15.0%
├── MEDIUM: 8.0% ≤ |P - 50%| < 15.0%
└── LOW:    |P - 50%| < 8.0%
Rationale:

55%/45% neutral zone: Prevents overtrading on weak signals
Weekly priority: Higher accuracy (67.4%) vs next-day (59.7%)
Edge-based strength: Quantifies signal quality
Raw probabilities: No artificial confidence inflation


📈 Model Performance
Validation Accuracy (Out-of-Sample)
MetricAccuracyvs Randomvs IndustryWeekly Direction67.4%+17.4%Above avg hedge fund (52-58%)Next-Day Direction59.7%+9.7%CompetitiveRandom Baseline50.0%--
Production Status: ✅ Ready for live trading with proper risk management

🛠️ Project Structure
stock-prediction/
├── train.py                 # Training pipeline
├── predict.py               # CLI prediction interface
├── config.py                # Hyperparameters & thresholds
├── src/
│   ├── data_loader.py       # EODHD API integration + caching
│   ├── feature_engineer.py  # Technical indicator computation
│   ├── model_builder.py     # Multi-task LSTM architecture
│   └── trainer.py           # Training & validation logic
├── models/
│   └── lstm_stock_model.h5  # Trained model (67.4% acc)
├── data/
│   └── raw/                 # Cached historical data
├── requirements.txt         # Python dependencies
└── README.md

📱 Batch Predictions
Analyze All 6 Stocks
Bash/Linux/Mac:
bashfor stock in AAPL MSFT NVDA AMZN GOOGL META; do
    python predict.py --stock $stock
done
PowerShell/Windows:
powershellforeach ($stock in @("AAPL","MSFT","NVDA","AMZN","GOOGL","META")) {
    python predict.py --stock $stock
}

🔧 Configuration
Customize in config.py
python# Model Hyperparameters
SEQUENCE_LENGTH = 60        # Days of history per prediction
LSTM_UNITS = [64, 32]       # Layer sizes
DENSE_UNITS = [64, 32]      # Dense layer sizes
DROPOUT_RATE = 0.2          # Regularization

# Signal Thresholds
BUY_THRESHOLD = 0.55        # 55% probability for BUY
SELL_THRESHOLD = 0.45       # 45% probability for SELL
HIGH_EDGE = 0.15            # 15% edge = HIGH confidence
MEDIUM_EDGE = 0.08          # 8% edge = MEDIUM confidence

# Training
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

🔮 Roadmap

 Regime Filter: Cross-validate signals with SPY/QQQ market direction
 Volatility-Based Sizing: ATR/VIX-adjusted position sizing
 Kelly Criterion: Optimal bet sizing based on edge strength
 Live Tracking Dashboard: Real-time P&L and rolling accuracy
 REST API: FastAPI backend with React frontend
 Ensemble Models: Combine LSTM with XGBoost/Transformer
 Options Integration: Implied volatility signals


⚠️ Risk Disclaimer
This software is for educational and research purposes only.

Past performance does not guarantee future results
67.4% accuracy means 32.6% of signals will be wrong
Always use proper risk management (stop-losses, position sizing)
Never risk more than 1-2% of capital per trade
Consult a licensed financial advisor before trading
Markets can remain irrational longer than you can stay solvent

The authors assume no liability for trading losses.

📦 Dependencies
txttensorflow>=2.10.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
ta>=0.10.0              # Technical analysis library
requests>=2.28.0
matplotlib>=3.6.0

🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add AmazingFeature')
Push to branch (git push origin feature/AmazingFeature)
Open a Pull Request


📄 License
MIT License - Free for personal and commercial use.
See LICENSE file for details.

🙏 Acknowledgments

EODHD for financial data API
TensorFlow team for deep learning framework
TA-Lib contributors for technical analysis tools

