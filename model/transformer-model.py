import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

# trading config
START_DATE = '2025-01-01'
END_DATE = '2025-12-24'
INITIAL_CAPITAL = 100000
REBALANCE_FREQUENCY = 'W'  #weekly, change to M for monthly
MIN_STOCKS = 2
MAX_STOCKS = 7
TAKE_PROFIT_THRESHOLD = 0.15  # max of 15% take profit
STOP_LOSS_THRESHOLD = -0.08   # 8% stop loss
MIN_PROBABILITY = 0.55         # minimum probability to enter position
MIN_HOLDING_DAYS = 5           # minimum days to hold before selling

print("\n" + "="*80)
print(" Portfolio Equity Simulation ")
print("="*80)
print(f"\nConfiguration:")
print(f"  Period: {START_DATE} to {END_DATE}")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"  Rebalancing: {REBALANCE_FREQUENCY}ly")
print(f"  Portfolio Size: {MIN_STOCKS}-{MAX_STOCKS} stocks")
print(f"  Take Profit: {TAKE_PROFIT_THRESHOLD*100:.1f}%")
print(f"  Stop Loss: {STOP_LOSS_THRESHOLD*100:.1f}%")

# Load data from screener csv
csv_path = os.path.join(os.path.dirname(__file__), '..', 'stock-pick', 'sp500_stock_picks.csv')
df = pd.read_csv(csv_path)
top_50_tickers = df.head(50)['Ticker'].tolist()
print(f"\nLoaded 50 candidate stocks from CSV")

print(f"\nDownloading historical data...")
stocks_data = {}
for ticker in top_50_tickers:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start='2024-09-01', end=END_DATE)  # Extra history for training
        if len(data) > 60:
            stocks_data[ticker] = data
            print(f"{ticker}: {len(data)} days")
    except:
        print(f"{ticker}: Failed")

print(f"\nDownloaded data for {len(stocks_data)} stocks")

# Download S&P 500 benchmark
spy = yf.Ticker('SPY')
spy_data = spy.history(start='2024-09-01', end=END_DATE)
print(f"S&P 500 benchmark: {len(spy_data)} days")

# Model architecture
class TransformerPredictor(nn.Module):
    def __init__(self, input_size=20, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 32)
        self.fc_mean = nn.Linear(32, 1)
        self.fc_std = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.fc_std(x)) + 0.001
        return mean, std

def train_model(returns, lookback=20, epochs=30):
    if len(returns) < lookback + 10:
        return None
    
    X, y = [], []
    for i in range(lookback, len(returns)):
        X.append(returns.iloc[i-lookback:i].values)
        y.append(returns.iloc[i])
    
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).unsqueeze(1)
    
    model = TransformerPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        mean_pred, std_pred = model(X)
        loss = torch.mean((y - mean_pred)**2 / (std_pred + 1e-6) + torch.log(std_pred + 1e-6))
        loss.backward()
        optimizer.step()
    
    return model

def predict_with_model(model, recent_returns):
    """make prediction based on test"""
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(recent_returns[-20:].values).unsqueeze(0)
        mean, std = model(X)
        return mean.item(), std.item()

# eval performance
def evaluate_stock(ticker, data, current_date):
    """Evaluate a stock using transformer and t-student distt to pick for trade"""
    # get data up to current date
    historical = data[data.index <= current_date]
    if len(historical) < 40:
        return None
    
    returns = historical['Close'].pct_change().dropna()
    if len(returns) < 30:
        return None
    
    # training
    model = train_model(returns, lookback=20, epochs=30)
    if model is None:
        return None
    
    # prediction
    mean_pred, std_pred = predict_with_model(model, returns)
    
    # T-distribution statistics
    df_freedom = len(returns) - 1
    t_stat = mean_pred / (std_pred + 1e-8)
    p_value = 1 - stats.t.cdf(abs(t_stat), df_freedom)
    probability = 1 - 2 * p_value if t_stat > 0 else 2 * p_value
    
    # Confidence interval
    t_critical = stats.t.ppf(0.975, df_freedom)
    ci_lower = mean_pred - t_critical * std_pred
    ci_upper = mean_pred + t_critical * std_pred
    
    # Additional metrics
    recent_vol = returns.tail(20).std()
    momentum = returns.tail(10).mean()
    sharpe = mean_pred / (std_pred + 1e-8) if std_pred > 0 else 0
    
    # Composite scoring
    composite_score = (
        probability * 0.35 +
        (sharpe if sharpe > 0 else 0) * 0.25 +
        (1 - recent_vol) * 0.20 +
        (momentum if momentum > 0 else 0) * 0.20
    )
    
    return {
        'Ticker': ticker,
        'Predicted_Return': mean_pred,
        'Uncertainty': std_pred,
        'T_Statistic': t_stat,
        'Probability': probability,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'Sharpe': sharpe,
        'Momentum': momentum,
        'Volatility': recent_vol,
        'Composite_Score': composite_score,
        'Current_Price': historical['Close'].iloc[-1]
    }

# portfolio mechanism
class DynamicPortfolioManager:
    def __init__(self, initial_capital, stocks_data, spy_data):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {ticker: {'shares': n, 'entry_price': p, 'entry_date': date}}
        self.stocks_data = stocks_data
        self.spy_data = spy_data
        self.portfolio_history = []
        self.trades_log = []
        
    def get_portfolio_value(self, date):
        """Calculate total portfolio value at given date"""
        total = self.cash
        for ticker, position in self.positions.items():
            if ticker in self.stocks_data:
                price_data = self.stocks_data[ticker]
                price_at_date = price_data[price_data.index <= date]
                if len(price_at_date) > 0:
                    current_price = price_at_date['Close'].iloc[-1]
                    total += position['shares'] * current_price
                    #total += position['shares'] * position['entry_price']
        return total
    
    def check_stop_loss_take_profit(self, date):
        """Check all positions for stop loss and take profit triggers"""
        to_sell = []
        
        for ticker, position in self.positions.items():
            if ticker not in self.stocks_data:
                continue
                
            price_data = self.stocks_data[ticker]
            current_price_data = price_data[price_data.index <= date]
            
            if len(current_price_data) == 0:
                continue
                
            current_price = current_price_data['Close'].iloc[-1]
            entry_price = position['entry_price']
            return_pct = (current_price - entry_price) / entry_price
            days_held = (date - position['entry_date']).days
            
            # Check triggers
            if return_pct >= TAKE_PROFIT_THRESHOLD and days_held >= MIN_HOLDING_DAYS:
                to_sell.append((ticker, 'TAKE_PROFIT', return_pct))
            elif return_pct <= STOP_LOSS_THRESHOLD:
                to_sell.append((ticker, 'STOP_LOSS', return_pct))
        
        # Execute sells
        for ticker, reason, return_pct in to_sell:
            self.sell_position(ticker, date, reason, return_pct)
    
    def sell_position(self, ticker, date, reason, return_pct):
        """Sell a position"""
        if ticker not in self.positions:
            return
            
        position = self.positions[ticker]
        price_data = self.stocks_data[ticker]
        current_price = price_data[price_data.index <= date]['Close'].iloc[-1]
        
        proceeds = position['shares'] * current_price
        self.cash += proceeds
        
        self.trades_log.append({
            'Date': date,
            'Action': 'SELL',
            'Ticker': ticker,
            'Shares': position['shares'],
            'Price': current_price,
            'Value': proceeds,
            'Return': return_pct,
            'Reason': reason
        })
        
        del self.positions[ticker]
    
    def rebalance(self, date, candidate_evaluations):
        """Rebalance portfolio based on new evaluations"""
        # Filter candidates
        valid_candidates = [
            c for c in candidate_evaluations 
            if c['Probability'] >= MIN_PROBABILITY and c['CI_Lower'] > 0
        ]
        
        # Sort by composite score
        valid_candidates.sort(key=lambda x: x['Composite_Score'], reverse=True)
        
        # Select top stocks
        selected = valid_candidates[:MAX_STOCKS]
        selected = [c for c in selected if c['Composite_Score'] > 0]
        
        if len(selected) < MIN_STOCKS and len(valid_candidates) >= MIN_STOCKS:
            selected = valid_candidates[:MIN_STOCKS]
        
        if len(selected) == 0:
            return
        
        selected_tickers = [s['Ticker'] for s in selected]
        
        # Sell positions not in new selection (after minimum holding period)
        for ticker in list(self.positions.keys()):
            days_held = (date - self.positions[ticker]['entry_date']).days
            if ticker not in selected_tickers and days_held >= MIN_HOLDING_DAYS:
                price_data = self.stocks_data[ticker]
                current_price = price_data[price_data.index <= date]['Close'].iloc[-1]
                return_pct = (current_price - self.positions[ticker]['entry_price']) / self.positions[ticker]['entry_price']
                self.sell_position(ticker, date, 'REBALANCE', return_pct)
        
        # Calculate target allocation
        total_value = self.get_portfolio_value(date)
        target_per_stock = total_value / len(selected)
        
        # Buy new positions or adjust existing
        for stock_eval in selected:
            ticker = stock_eval['Ticker']
            current_price = stock_eval['Current_Price']
            
            if ticker not in self.positions:
                # New position
                if self.cash > target_per_stock:
                    shares = int(target_per_stock / current_price)
                    cost = shares * current_price
                    
                    if shares > 0 and self.cash >= cost:
                        self.cash -= cost
                        self.positions[ticker] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_date': date
                        }
                        
                        self.trades_log.append({
                            'Date': date,
                            'Action': 'BUY',
                            'Ticker': ticker,
                            'Shares': shares,
                            'Price': current_price,
                            'Value': cost,
                            'Return': 0,
                            'Reason': 'NEW_POSITION'
                        })
    
    def run_backtest(self, start_date, end_date):
        """Run portfolio backtest"""
        print("\n" + "="*80)
        print("RUNNING BACKTEST")
        print("="*80)
        
        # Convert string dates to datetime to prevent bug
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Get trading dates from SPY data
        spy_dates = self.spy_data.index
        
        # Make dates timezone-aware if SPY data has timezone
        if spy_dates.tz is not None:
            if start_date.tz is None:
                start_date = start_date.tz_localize(spy_dates.tz)
            if end_date.tz is None:
                end_date = end_date.tz_localize(spy_dates.tz)
        
        trading_dates = [d for d in spy_dates if d >= start_date and d <= end_date]
        
        # Get rebalance dates
        if REBALANCE_FREQUENCY == 'W':
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        else:
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        rebalance_dates = [d for d in rebalance_dates if d in trading_dates]
        
        # Also evaluate stocks during the period
        available_tickers = list(self.stocks_data.keys())
        
        print(f"\nTrading days: {len(trading_dates)}")
        print(f"Rebalance dates: {len(rebalance_dates)}")
        
        for date in trading_dates:
            # Check stop loss and take profit daily
            self.check_stop_loss_take_profit(date)
            
            # Rebalance on schedule
            if date in rebalance_dates:
                print(f"\n[{date.strftime('%Y-%m-%d')}] Rebalancing")
                
                # Evaluate all stocks
                evaluations = []
                for ticker in available_tickers:
                    eval_result = evaluate_stock(ticker, self.stocks_data[ticker], date)
                    if eval_result:
                        evaluations.append(eval_result)
                
                print(f"Evaluated {len(evaluations)} stocks")
                
                # Rebalancing update
                self.rebalance(date, evaluations)
                
                print(f"  Positions: {len(self.positions)}")
                print(f"  Cash: ${self.cash:,.0f}")
                print(f"  Portfolio Value: ${self.get_portfolio_value(date):,.0f}")
            
            # Record daily portfolio value
            portfolio_value = self.get_portfolio_value(date)
            spy_price = self.spy_data[self.spy_data.index <= date]['Close'].iloc[-1]
            
            self.portfolio_history.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Cash': self.cash,
                'Positions': len(self.positions),
                'SPY_Price': spy_price
            })

# Run backtest
manager = DynamicPortfolioManager(INITIAL_CAPITAL, stocks_data, spy_data)
manager.run_backtest(START_DATE, END_DATE)

# acalculate performance metrics
print("\n" + "="*80)
print("CALCULATING PERFORMANCE METRICS")
print("="*80)

portfolio_df = pd.DataFrame(manager.portfolio_history)

if len(portfolio_df) == 0:
    print("\nERROR: No portfolio history generated. Check date ranges.")
    print(f"START_DATE: {START_DATE}")
    print(f"END_DATE: {END_DATE}")
    print(f"SPY data range: {spy_data.index[0]} to {spy_data.index[-1]}")
    exit(1)

portfolio_df.set_index('Date', inplace=True)

# Portfolio returns
portfolio_df['Portfolio_Return'] = portfolio_df['Portfolio_Value'].pct_change()

# Benchmark returns
spy_start = spy_data[spy_data.index >= START_DATE]
spy_returns = spy_start['Close'].pct_change().dropna()

# Align data
common_idx = portfolio_df.index.intersection(spy_returns.index)
portfolio_returns = portfolio_df.loc[common_idx, 'Portfolio_Return'].dropna()
spy_returns_aligned = spy_returns.loc[common_idx]

# Calculate metrics
def calculate_metrics(returns):
    cum_ret = (1 + returns).prod() - 1
    
    # Sortino
    downside = returns[returns < 0]
    sortino = (returns.mean() / np.sqrt((downside**2).mean()) * np.sqrt(252)) if len(downside) > 0 else 999
    
    # Max Drawdown
    cum = (1 + returns).cumprod()
    running_max = cum.expanding().max()
    drawdown = ((cum - running_max) / running_max).min()
    
    # Sharpe
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    
    return {
        'Cumulative_Return': cum_ret,
        'Sortino_Ratio': sortino,
        'Max_Drawdown': drawdown,
        'Sharpe_Ratio': sharpe,
        'Volatility': vol
    }

port_metrics = calculate_metrics(portfolio_returns)
bench_metrics = calculate_metrics(spy_returns_aligned)

# Print results
print("\n" + "="*80)
print("PERFORMANCE REPORT")
print("="*80)

print("\nPORTFOLIO PERFORMANCE:")
for metric, value in port_metrics.items():
    print(f"  {metric:20s}: {value:>12.4f}")

print("\nBENCHMARK PERFORMANCE (S&P 500):")
for metric, value in bench_metrics.items():
    print(f"  {metric:20s}: {value:>12.4f}")

print("\nRELATIVE PERFORMANCE:")
print(f"  Excess Return:        {(port_metrics['Cumulative_Return'] - bench_metrics['Cumulative_Return']):>12.4f}")
print(f"  Sortino Difference:   {(port_metrics['Sortino_Ratio'] - bench_metrics['Sortino_Ratio']):>12.4f}")

# Trading activity
print("\n" + "="*80)
print("TRADING ACTIVITY")
print("="*80)

trades_df = pd.DataFrame(manager.trades_log)
if len(trades_df) > 0:
    print(f"\nTotal Trades: {len(trades_df)}")
    print(f"  Buys:  {len(trades_df[trades_df['Action'] == 'BUY'])}")
    print(f"  Sells: {len(trades_df[trades_df['Action'] == 'SELL'])}")
    
    sells = trades_df[trades_df['Action'] == 'SELL']
    if len(sells) > 0:
        print(f"\nSell Reasons:")
        for reason in sells['Reason'].unique():
            count = len(sells[sells['Reason'] == reason])
            avg_return = sells[sells['Reason'] == reason]['Return'].mean()
            print(f"  {reason:15s}: {count:3d} trades, Avg Return: {avg_return:>8.2%}")
        
        print(f"\nAverage Return per Trade: {sells['Return'].mean():.2%}")
        print(f"Win Rate: {len(sells[sells['Return'] > 0]) / len(sells):.2%}")

# Final portfolio composition
print("\n" + "="*80)
print("FINAL PORTFOLIO")
print("="*80)

if len(manager.positions) > 0:
    final_date = portfolio_df.index[-1]
    final_value = manager.get_portfolio_value(final_date)
    
    final_positions = []
    for ticker, position in manager.positions.items():
        current_price = stocks_data[ticker][stocks_data[ticker].index <= final_date]['Close'].iloc[-1]
        value = position['shares'] * current_price
        weight = value / final_value
        return_pct = (current_price - position['entry_price']) / position['entry_price']
        
        final_positions.append({
            'Ticker': ticker,
            'Shares': position['shares'],
            'Entry_Price': position['entry_price'],
            'Current_Price': current_price,
            'Value': value,
            'Weight': weight,
            'Return': return_pct
        })
    
    final_df = pd.DataFrame(final_positions)
    print(f"\nPositions: {len(final_df)}")
    print(f"Cash: ${manager.cash:,.2f}")
    print(f"Total Value: ${final_value:,.2f}")
    print("\n", final_df.to_string(index=False))
else:
    print("\nNo positions held at end of period")
    print(f"Cash: ${manager.cash:,.2f}")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Portfolio composition
if len(manager.positions) > 0:
    portfolio_comp = pd.DataFrame([
        {'Ticker': ticker, 'Weight': (position['shares'] * stocks_data[ticker]['Close'].iloc[-1]) / manager.get_portfolio_value(portfolio_df.index[-1])}
        for ticker, position in manager.positions.items()
    ])
else:
    portfolio_comp = pd.DataFrame({'Ticker': ['CASH'], 'Weight': [1.0]})

portfolio_comp.to_csv('c:\\Users\\seva\\transformer-mr-model\\portfolio_composition.csv', index=False)

# Performance metrics
metrics_df = pd.DataFrame({
    'Metric': list(port_metrics.keys()),
    'Portfolio': list(port_metrics.values()),
    'Benchmark': list(bench_metrics.values())
})
metrics_df.to_csv('c:\\Users\\seva\\transformer-mr-model\\performance_metrics.csv', index=False)

# Trades log
if len(trades_df) > 0:
    trades_df.to_csv('c:\\Users\\seva\\transformer-mr-model\\trades_log.csv', index=False)

# Portfolio history
portfolio_df.to_csv('c:\\Users\\seva\\transformer-mr-model\\portfolio_history.csv')

print("\nSaved: portfolio_composition.csv")
print("Saved: performance_metrics.csv")
print("Saved: trades_log.csv")
print("Saved: portfolio_history.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print()
