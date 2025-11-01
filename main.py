import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"


@dataclass
class MarketData:
    """Represents a single tick of market data"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int = 1000


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    order_type: OrderType
    quantity: int
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_percent(self) -> float:
        return (self.unrealized_pnl / self.cost_basis) * 100 if self.cost_basis != 0 else 0.0


@dataclass
class Trade:
    """Represents a completed trade"""
    trade_id: str
    symbol: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    realized_pnl: float
    realized_pnl_percent: float


class MarketDataGenerator:
    """Generates synthetic market data using geometric Brownian motion with stochastic volatility"""
    
    def __init__(self, symbols: List[str], initial_prices: Dict[str, float],
                 mu: float = 0.0001, initial_sigma: float = 0.02,
                 sigma: Optional[float] = None,
                 kappa: float = 3.0, theta: float = 0.02, xi: float = 0.3,
                 rho: float = -0.7, seed: Optional[int] = None):
        """
        Initialize market data generator with stochastic volatility (Heston-like model)
        
        Args:
            symbols: List of asset symbols
            initial_prices: Dictionary of initial prices for each symbol
            mu: Drift coefficient (trend) for price
            initial_sigma: Initial volatility level
            kappa: Mean reversion speed for volatility
            theta: Long-term mean volatility level
            xi: Volatility of volatility (vol of vol)
            rho: Correlation between price and volatility shocks (-1 to 1)
            seed: Random seed for reproducibility
        """
        self.symbols = symbols
        self.current_prices = initial_prices.copy()
        # Accept either `initial_sigma` or the legacy/alternate name `sigma`.
        chosen_initial_sigma = float(sigma) if sigma is not None else float(initial_sigma)
        self.current_volatilities = {symbol: chosen_initial_sigma for symbol in symbols}

        # Price dynamics parameters
        self.mu = mu
        
        # Volatility dynamics parameters (Heston model)
        self.kappa = kappa  # Speed of mean reversion
        self.theta = theta  # Long-term volatility mean
        self.xi = xi        # Volatility of volatility
        self.rho = rho      # Correlation between price and vol shocks
        
        if seed is not None:
            np.random.seed(seed)
        self.tick_count = 0
        
        # Store volatility history for analysis
        self.volatility_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
    
    def generate_tick(self, timestamp: datetime) -> List[MarketData]:
        """Generate a single tick of data for all symbols using stochastic volatility"""
        ticks = []
        dt = 1.0  # Time step (can be adjusted for different frequencies)
        
        for symbol in self.symbols:
            # Generate correlated random shocks
            # Using Cholesky decomposition for correlation
            z1 = np.random.normal(0, 1)
            z2 = np.random.normal(0, 1)
            
            # Correlated shock for price
            w_price = z1
            # Correlated shock for volatility
            w_vol = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
            
            # Current volatility (ensure non-negative)
            current_vol = max(0.001, self.current_volatilities[symbol])
            
            # Update volatility using mean-reverting process (CIR-like process)
            # dσ² = κ(θ - σ²)dt + ξσdW
            vol_drift = self.kappa * (self.theta - current_vol) * dt
            vol_diffusion = self.xi * np.sqrt(current_vol) * np.sqrt(dt) * w_vol
            
            new_volatility = current_vol + vol_drift + vol_diffusion
            # Apply Feller condition to prevent negative volatility
            new_volatility = max(0.001, new_volatility)
            self.current_volatilities[symbol] = new_volatility
            
            # Store volatility history
            self.volatility_history[symbol].append(new_volatility)
            
            # Update price using GBM with stochastic volatility
            # dS = μS dt + σS dW
            price_drift = self.mu * self.current_prices[symbol] * dt
            price_diffusion = current_vol * self.current_prices[symbol] * np.sqrt(dt) * w_price
            
            price_change = price_drift + price_diffusion
            new_price = self.current_prices[symbol] + price_change
            
            # Ensure price stays positive
            self.current_prices[symbol] = max(0.01, new_price)
            
            ticks.append(MarketData(
                timestamp=timestamp,
                symbol=symbol,
                price=round(self.current_prices[symbol], 2),
                volume=np.random.randint(500, 2000)
            ))
        
        self.tick_count += 1
        return ticks
    
    def get_current_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol"""
        return self.current_volatilities.get(symbol, 0.0)
    
    def get_volatility_stats(self, symbol: str) -> Dict:
        """Get volatility statistics for a symbol"""
        if symbol not in self.volatility_history or not self.volatility_history[symbol]:
            return {}
        
        vol_data = self.volatility_history[symbol]
        return {
            'current': vol_data[-1],
            'mean': np.mean(vol_data),
            'std': np.std(vol_data),
            'min': np.min(vol_data),
            'max': np.max(vol_data)
        }
    
    def generate_batch(self, start_time: datetime, periods: int, 
                      freq: timedelta = timedelta(seconds=1)) -> pd.DataFrame:
        """Generate batch data for backtesting"""
        all_data = []
        current_time = start_time
        
        for _ in range(periods):
            ticks = self.generate_tick(current_time)
            for tick in ticks:
                all_data.append({
                    'timestamp': tick.timestamp,
                    'symbol': tick.symbol,
                    'price': tick.price,
                    'volume': tick.volume
                })
            current_time += freq
        
        return pd.DataFrame(all_data)


class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, market_data: pd.DataFrame, portfolio) -> List[Dict]:
        """Generate trading signals based on market data"""
        raise NotImplementedError


class MovingAverageCrossover(TradingStrategy):
    """Simple moving average crossover strategy"""
    
    def __init__(self, short_window: int = 10, long_window: int = 30, position_size: int = 100):
        super().__init__("MA_Crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
    
    def calculate_position_size(self, portfolio) -> int:
        """Calculate dynamic position size based on portfolio value ratio"""
        portfolio_ratio = portfolio.total_portfolio_value / portfolio.initial_cash
        dynamic_size = max(1, int(self.position_size * portfolio_ratio))
        return dynamic_size
    
    def generate_signals(self, market_data: pd.DataFrame, portfolio) -> List[Dict]:
        """Generate buy/sell signals based on MA crossover"""
        signals = []
        
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < self.long_window:
                continue
            
            # Calculate moving averages
            symbol_data['MA_short'] = symbol_data['price'].rolling(window=self.short_window).mean()
            symbol_data['MA_long'] = symbol_data['price'].rolling(window=self.long_window).mean()
            
            # Get latest values
            latest = symbol_data.iloc[-1]
            previous = symbol_data.iloc[-2] if len(symbol_data) > 1 else None
            
            if previous is None or pd.isna(latest['MA_short']) or pd.isna(latest['MA_long']):
                continue
            
            # Check for crossover
            current_position = portfolio.get_position(symbol)
            
            # Calculate dynamic position size based on current portfolio value
            dynamic_size = self.calculate_position_size(portfolio)
            
            # Bullish crossover: short MA crosses above long MA
            if (previous['MA_short'] <= previous['MA_long'] and 
                latest['MA_short'] > latest['MA_long'] and 
                current_position is None):
                signals.append({
                    'symbol': symbol,
                    'action': OrderType.BUY,
                    'quantity': dynamic_size,
                    'price': latest['price'],
                    'timestamp': latest['timestamp']
                })
            
            # Bearish crossover: short MA crosses below long MA
            elif (previous['MA_short'] >= previous['MA_long'] and 
                  latest['MA_short'] < latest['MA_long'] and 
                  current_position is not None):
                signals.append({
                    'symbol': symbol,
                    'action': OrderType.SELL,
                    'quantity': current_position.quantity,
                    'price': latest['price'],
                    'timestamp': latest['timestamp']
                })
        
        return signals


class Portfolio:
    """Manages portfolio positions, cash, and P&L tracking"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.orders: List[Order] = []
        self.trade_counter = 0
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol"""
        return self.positions.get(symbol)
    
    def update_prices(self, market_data: List[MarketData]):
        """Update current prices for all positions"""
        for tick in market_data:
            if tick.symbol in self.positions:
                self.positions[tick.symbol].current_price = tick.price
    
    def execute_order(self, order: Order) -> bool:
        """Execute a trading order"""
        if order.order_type == OrderType.BUY:
            cost = order.quantity * order.price
            if cost > self.cash:
                print(f"Insufficient funds for {order.symbol}: Need ${cost:.2f}, Have ${self.cash:.2f}")
                return False
            
            self.cash -= cost
            
            if order.symbol in self.positions:
                # Average down/up existing position
                pos = self.positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                avg_price = ((pos.quantity * pos.entry_price) + (order.quantity * order.price)) / total_quantity
                pos.quantity = total_quantity
                pos.entry_price = avg_price
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=order.price,
                    entry_time=order.timestamp,
                    current_price=order.price
                )
            
            order.status = OrderStatus.EXECUTED
            order.execution_price = order.price
            order.execution_time = order.timestamp
            return True
        
        elif order.order_type == OrderType.SELL:
            if order.symbol not in self.positions:
                print(f"No position to sell for {order.symbol}")
                return False
            
            pos = self.positions[order.symbol]
            if order.quantity > pos.quantity:
                print(f"Insufficient shares for {order.symbol}: Need {order.quantity}, Have {pos.quantity}")
                return False
            
            # Calculate realized P&L
            proceeds = order.quantity * order.price
            cost_basis = order.quantity * pos.entry_price
            realized_pnl = proceeds - cost_basis
            realized_pnl_pct = (realized_pnl / cost_basis) * 100 if cost_basis != 0 else 0.0
            
            self.cash += proceeds
            
            # Record trade
            self.trade_counter += 1
            trade = Trade(
                trade_id=f"T{self.trade_counter:05d}",
                symbol=order.symbol,
                quantity=order.quantity,
                entry_price=pos.entry_price,
                exit_price=order.price,
                entry_time=pos.entry_time,
                exit_time=order.timestamp,
                realized_pnl=realized_pnl,
                realized_pnl_percent=realized_pnl_pct
            )
            self.closed_trades.append(trade)
            
            # Update or close position
            pos.quantity -= order.quantity
            if pos.quantity == 0:
                del self.positions[order.symbol]
            
            order.status = OrderStatus.EXECUTED
            order.execution_price = order.price
            order.execution_time = order.timestamp
            return True
        
        return False
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        """Calculate total realized P&L from closed trades"""
        return sum(trade.realized_pnl for trade in self.closed_trades)
    
    @property
    def total_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_exposure(self) -> float:
        """Calculate total market exposure"""
        return sum(pos.market_value for pos in self.positions.values())
    
    def get_summary(self) -> Dict:
        """Get portfolio summary statistics"""
        return {
            'cash': self.cash,
            'total_portfolio_value': self.total_portfolio_value,
            'total_exposure': self.total_exposure,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_realized_pnl': self.total_realized_pnl,
            'total_pnl': self.total_unrealized_pnl + self.total_realized_pnl,
            'return_pct': ((self.total_portfolio_value - self.initial_cash) / self.initial_cash) * 100,
            'num_open_positions': len(self.positions),
            'num_closed_trades': len(self.closed_trades)
        }


class TradingEngine:
    """Main trading engine that orchestrates data, strategy, and portfolio"""
    
    def __init__(self, portfolio: Portfolio, strategy: TradingStrategy):
        self.portfolio = portfolio
        self.strategy = strategy
        self.order_counter = 0
        self.market_data_history: List[MarketData] = []
    
    def process_market_data(self, market_data: List[MarketData], historical_df: pd.DataFrame):
        """Process incoming market data and execute strategy"""
        # Update portfolio with latest prices
        self.portfolio.update_prices(market_data)
        self.market_data_history.extend(market_data)
        
        # Generate trading signals
        signals = self.strategy.generate_signals(historical_df, self.portfolio)
        
        # Execute orders based on signals
        for signal in signals:
            self.order_counter += 1
            order = Order(
                order_id=f"O{self.order_counter:05d}",
                symbol=signal['symbol'],
                order_type=signal['action'],
                quantity=signal['quantity'],
                price=signal['price'],
                timestamp=signal['timestamp']
            )
            
            if self.portfolio.execute_order(order):
                self.portfolio.orders.append(order)
    
    def print_portfolio_summary(self):
        """Print detailed portfolio summary"""
        summary = self.portfolio.get_summary()
        
        print("\n" + "="*80)
        print("PORTFOLIO SUMMARY")
        print("="*80)
        print(f"Cash:                    ${summary['cash']:,.2f}")
        print(f"Total Portfolio Value:   ${summary['total_portfolio_value']:,.2f}")
        print(f"Total Exposure:          ${summary['total_exposure']:,.2f}")
        print(f"Total Return:            {summary['return_pct']:.2f}%")
        print(f"\nUnrealized P&L:          ${summary['total_unrealized_pnl']:,.2f}")
        print(f"Realized P&L:            ${summary['total_realized_pnl']:,.2f}")
        print(f"Total P&L:               ${summary['total_pnl']:,.2f}")
        print(f"\nOpen Positions:          {summary['num_open_positions']}")
        print(f"Closed Trades:           {summary['num_closed_trades']}")
        
        if self.portfolio.positions:
            print("\n" + "-"*80)
            print("OPEN POSITIONS")
            print("-"*80)
            print(f"{'Symbol':<10} {'Qty':<8} {'Entry':<12} {'Current':<12} {'Value':<15} {'Unrealized P&L':<20}")
            print("-"*80)
            
            for symbol, pos in self.portfolio.positions.items():
                print(f"{symbol:<10} {pos.quantity:<8} ${pos.entry_price:<11.2f} "
                      f"${pos.current_price:<11.2f} ${pos.market_value:<14,.2f} "
                      f"${pos.unrealized_pnl:<10,.2f} ({pos.unrealized_pnl_percent:>6.2f}%)")
        
        if self.portfolio.closed_trades:
            print("\n" + "-"*80)
            print(f"CLOSED TRADES (Last 10)")
            print("-"*80)
            print(f"{'Trade ID':<10} {'Symbol':<10} {'Qty':<8} {'Entry':<12} {'Exit':<12} {'Realized P&L':<20}")
            print("-"*80)
            
            for trade in self.portfolio.closed_trades[-10:]:
                print(f"{trade.trade_id:<10} {trade.symbol:<10} {trade.quantity:<8} "
                      f"${trade.entry_price:<11.2f} ${trade.exit_price:<11.2f} "
                      f"${trade.realized_pnl:<10,.2f} ({trade.realized_pnl_percent:>6.2f}%)")
        
        print("="*80 + "\n")


def main():
    """Main simulation runner"""
    print("="*80)
    print("TRADING SIMULATION PLATFORM")
    print("="*80)
    
    # Initialize market data generator
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    initial_prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0}
    
    data_generator = MarketDataGenerator(
        symbols=symbols,
        initial_prices=initial_prices,
        mu=0.0002,  # Slight upward drift
        sigma=0.015,  # Moderate volatility
        seed=42
    )
    
    # Initialize portfolio and strategy
    portfolio = Portfolio(initial_cash=100000.0)
    strategy = MovingAverageCrossover(short_window=10, long_window=30, position_size=10)
    
    # Initialize trading engine
    engine = TradingEngine(portfolio=portfolio, strategy=strategy)
    
    # Generate historical data for initial strategy calculation
    start_time = datetime(2025, 1, 1, 9, 30, 0)
    print(f"\nGenerating market data starting from {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Strategy: {strategy.name} (Short MA: {strategy.short_window}, Long MA: {strategy.long_window})")
    print(f"Initial Capital: ${portfolio.initial_cash:,.2f}\n")
    
    # Run simulation for 1000 ticks (can be adjusted)
    num_ticks = 1000
    batch_data = data_generator.generate_batch(
        start_time=start_time,
        periods=num_ticks,
        freq=timedelta(minutes=1)  # 1-minute bars
    )
    
    # Process data tick by tick
    print("Running simulation...")
    report_interval = 200  # Print summary every 200 ticks
    
    for i in range(len(batch_data)):
        current_batch = batch_data.iloc[:i+1]
        latest_ticks = []
        
        for symbol in symbols:
            symbol_data = current_batch[current_batch['symbol'] == symbol]
            if not symbol_data.empty:
                latest = symbol_data.iloc[-1]
                latest_ticks.append(MarketData(
                    timestamp=latest['timestamp'],
                    symbol=latest['symbol'],
                    price=latest['price'],
                    volume=latest['volume']
                ))
        
        engine.process_market_data(latest_ticks, current_batch)
        
        # Print periodic updates
        if (i + 1) % report_interval == 0 or i == len(batch_data) - 1:
            print(f"\nTick {i+1}/{num_ticks} - {latest_ticks[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            engine.print_portfolio_summary()
    
    print("\nSimulation completed successfully!")
    print(f"Total ticks processed: {num_ticks}")


if __name__ == "__main__":
    main()