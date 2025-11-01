# Trading Simulation Platform

A comprehensive Python-based algorithmic trading simulation system with stochastic volatility modeling, strategy execution, and real-time portfolio tracking.

## 🎯 Overview

This project implements a fully self-contained trading simulation platform that demonstrates advanced concepts in algorithmic trading, quantitative finance, and risk management. The system generates realistic market data with stochastic volatility, executes trading strategies, and provides detailed performance analytics.

## ✨ Key Features

### 1. **Advanced Market Data Generation**
- **Stochastic Volatility Model** (Heston-like): Volatility follows a mean-reverting process
- **Geometric Brownian Motion** for price dynamics with time-varying volatility
- Configurable parameters:
  - `μ` (mu): Drift coefficient (price trend)
  - `σ` (sigma): Initial volatility level
  - `κ` (kappa): Mean reversion speed for volatility
  - `θ` (theta): Long-term mean volatility
  - `ξ` (xi): Volatility of volatility (vol of vol)
  - `ρ` (rho): Correlation between price and volatility shocks
- Support for multiple asset symbols simultaneously
- Both streaming and batch data generation modes

### 2. **Trading Strategy Engine**
- **Moving Average Crossover Strategy** (configurable windows)
- Automatic signal generation based on technical indicators
- Extensible architecture for adding custom strategies
- Order validation and execution logic
- Position sizing and risk controls

### 3. **Portfolio & Risk Management**
- **Real-time P&L Tracking**:
  - Unrealized P&L (open positions)
  - Realized P&L (closed trades)
  - Total return percentage
- **Position Management**:
  - Multi-asset portfolio support
  - Position averaging for multiple entries
  - Automatic position sizing
- **Risk Metrics**:
  - Total portfolio exposure
  - Per-asset exposure tracking
  - Cash balance management
- **Trade History**:
  - Complete trade log with entry/exit details
  - Performance analytics per trade
  - Order execution history

## 🏗️ Architecture

```
trading_platform.py
├── Market Data Layer
│   └── MarketDataGenerator (Stochastic volatility model)
├── Strategy Layer
│   ├── TradingStrategy (Base class)
│   └── MovingAverageCrossover (Implementation)
├── Execution Layer
│   ├── TradingEngine (Orchestration)
│   └── Order Management
└── Portfolio Layer
    ├── Portfolio (Position tracking)
    ├── Position (Individual holdings)
    └── Trade (Completed transactions)
```

## 📋 Requirements

```bash
numpy>=1.21.0
pandas>=1.3.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-simulation-platform.git
cd trading-simulation-platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

Run the simulation with default parameters:

```bash
python trading_platform.py
```

### Customizing Parameters

Edit the `main()` function to customize:

```python
# Market data parameters
data_generator = MarketDataGenerator(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    initial_prices={'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0},
    mu=0.0002,          # Price drift
    initial_sigma=0.02, # Initial volatility (2%)
    kappa=3.0,          # Volatility mean reversion speed
    theta=0.02,         # Long-term volatility mean
    xi=0.3,             # Volatility of volatility
    rho=-0.7,           # Price-volatility correlation
    seed=42
)

# Strategy parameters
strategy = MovingAverageCrossover(
    short_window=10,    # Short MA period
    long_window=30,     # Long MA period
    position_size=10    # Shares per trade
)

# Portfolio parameters
portfolio = Portfolio(initial_cash=100000.0)
```

### Example Output

```
================================================================================
TRADING SIMULATION PLATFORM
================================================================================

Generating market data starting from 2025-01-01 09:30:00
Symbols: AAPL, GOOGL, MSFT
Strategy: MA_Crossover (Short MA: 10, Long MA: 30)
Initial Capital: $100,000.00

Stochastic Volatility Parameters:
  Initial σ: 0.0200
  Mean reversion speed (κ): 3.0
  Long-term mean (θ): 0.02
  Vol of vol (ξ): 0.3
  Correlation (ρ): -0.7

Running simulation...

Tick 200/1000 - 2025-01-01 12:49:00

Current Volatility Levels:
  AAPL: σ=0.0215 (mean=0.0198, min=0.0145, max=0.0287)
  GOOGL: σ=0.0189 (mean=0.0203, min=0.0156, max=0.0245)
  MSFT: σ=0.0234 (mean=0.0211, min=0.0167, max=0.0298)

================================================================================
PORTFOLIO SUMMARY
================================================================================
Cash:                    $99,547.20
Total Portfolio Value:   $100,245.80
Total Exposure:          $698.60
Total Return:            0.25%

Unrealized P&L:          $45.80
Realized P&L:            -$200.00
Total P&L:               -$154.20

Open Positions:          1
Closed Trades:           2
```

## 🧪 Testing

Run the simulation with different parameters to test various scenarios:

### High Volatility Scenario
```python
data_generator = MarketDataGenerator(
    ...,
    xi=0.5,  # Higher vol of vol
    theta=0.04  # Higher long-term volatility
)
```

### Mean-Reverting Market
```python
data_generator = MarketDataGenerator(
    ...,
    mu=-0.0001,  # Negative drift
    kappa=5.0  # Faster mean reversion
)
```

## 📊 Output Metrics

### Portfolio Metrics
- **Cash**: Available liquid capital
- **Total Portfolio Value**: Cash + market value of positions
- **Total Exposure**: Market value of all open positions
- **Total Return**: Percentage return from initial capital

### P&L Metrics
- **Unrealized P&L**: Profit/loss on open positions (mark-to-market)
- **Realized P&L**: Profit/loss from closed trades
- **Total P&L**: Sum of unrealized and realized P&L

### Position Details
- Symbol, quantity, entry price, current price
- Market value and unrealized P&L per position
- Percentage gain/loss per position

### Trade History
- Trade ID, symbol, quantity
- Entry and exit prices
- Realized P&L and percentage return per trade

## 🔧 Extending the Platform

### Adding a New Strategy

Create a new strategy by inheriting from `TradingStrategy`:

```python
class MomentumStrategy(TradingStrategy):
    def __init__(self, lookback_period: int = 20, threshold: float = 0.02):
        super().__init__("Momentum")
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def generate_signals(self, market_data: pd.DataFrame, portfolio) -> List[Dict]:
        signals = []
        # Implement your strategy logic here
        return signals
```

### Adding Custom Metrics

Extend the `Portfolio` class to track additional metrics:

```python
@property
def sharpe_ratio(self) -> float:
    """Calculate Sharpe ratio"""
    # Implementation here
    pass
```

## 📈 Performance Considerations

- The simulation processes data tick-by-tick for accuracy
- Batch data generation is used for efficiency
- Memory usage scales with number of ticks and symbols
- For longer simulations (>10,000 ticks), consider periodic data cleanup

## 🎓 Educational Value

This project demonstrates:
- **Quantitative Finance**: Stochastic processes, volatility modeling
- **Algorithmic Trading**: Strategy design, signal generation, execution
- **Risk Management**: Portfolio tracking, P&L calculation, exposure management
- **Software Engineering**: OOP design, modular architecture, extensibility
- **Python**: NumPy, Pandas, dataclasses, type hints

## 📝 Future Enhancements

- [ ] Multiple strategy support (portfolio of strategies)
- [ ] Advanced risk metrics (VaR, Sharpe ratio, max drawdown)
- [ ] Real-time visualization (matplotlib/plotly)
- [ ] Backtesting framework with historical data
- [ ] Order types (limit, stop-loss, trailing stop)
- [ ] Transaction costs and slippage modeling
- [ ] Multi-threaded execution for real-time simulation
- [ ] ML-based strategy optimization


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

[Your Name]
- GitHub: [@l0ller](https://github.com/l0ller)
