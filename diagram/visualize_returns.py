import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
START_DATE = '2025-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

print("Loading portfolio composition")
portfolio = pd.read_csv('portfolio_composition.csv')
print(f"\nPortfolio Holdings:")
print(portfolio)

# Download historical data for portfolio stocks
print(f"\nDownloading portfolio data from {START_DATE} to {END_DATE}...")
portfolio_returns = []

for _, row in portfolio.iterrows():
    ticker = row['Ticker']
    weight = row['Weight']
    
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if not data.empty:
            # Handle both single-ticker and multi-ticker dataframes
            if 'Adj Close' in data.columns:
                close_data = data['Adj Close']
            elif ('Adj Close', ticker) in data.columns:
                close_data = data[('Adj Close', ticker)]
            else:
                # For single ticker, columns might be at top level
                close_data = data['Close'] if 'Close' in data.columns else data.iloc[:, data.columns.get_loc('Close')]
            
            returns = close_data.pct_change()
            weighted_returns = returns * weight
            portfolio_returns.append(weighted_returns)
            print(f"{ticker}: {len(data)} days")
    except Exception as e:
        print(f"{ticker}: Error - {e}")

# Combine portfolio returns
print("\nCalculating portfolio performance...")
portfolio_returns_series = pd.concat(portfolio_returns, axis=1).sum(axis=1)

# Download S&P 500 benchmark
print("Downloading S&P 500 benchmark data...")
spy_data = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
if 'Adj Close' in spy_data.columns:
    benchmark_returns = spy_data['Adj Close'].pct_change()
elif 'Close' in spy_data.columns:
    benchmark_returns = spy_data['Close'].pct_change()
else:
    benchmark_returns = spy_data.iloc[:, 3].pct_change()  # Usually 4th column is close

# Align dates
common_dates = portfolio_returns_series.index.intersection(benchmark_returns.index)
portfolio_returns_series = portfolio_returns_series.loc[common_dates]
benchmark_returns = benchmark_returns.loc[common_dates]

# Calculate cumulative returns
portfolio_cumulative = (1 + portfolio_returns_series).cumprod()
benchmark_cumulative = (1 + benchmark_returns).cumprod()

# Create visualization
print("\nCreating visualization...")
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Transformer Model Portfolio Performance Analysis', fontsize=16, fontweight='bold')

# 1. Cumulative Returns
ax1 = axes[0]
ax1.plot(portfolio_cumulative.index, (portfolio_cumulative - 1) * 100, 
         label='Transformer Portfolio', linewidth=2, color='#2E86AB')
ax1.plot(benchmark_cumulative.index, (benchmark_cumulative - 1) * 100, 
         label='S&P 500 (SPY)', linewidth=2, color='#A23B72', alpha=0.7)
ax1.set_title('Cumulative Returns Over Time', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cumulative Return (%)', fontsize=10)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

# Add performance metrics to the chart
final_portfolio_return = float((portfolio_cumulative.iloc[-1] - 1) * 100)
final_benchmark_return = float((benchmark_cumulative.iloc[-1] - 1) * 100)
outperformance = final_portfolio_return - final_benchmark_return

textstr = f'Portfolio: {final_portfolio_return:.2f}%\nBenchmark: {final_benchmark_return:.2f}%\nOutperformance: {outperformance:.2f}%'
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Daily Returns
ax2 = axes[1]
ax2.plot(portfolio_returns_series.index, portfolio_returns_series * 100, 
         label='Portfolio Daily Returns', linewidth=0.8, color='#2E86AB', alpha=0.6)
ax2.plot(benchmark_returns.index, benchmark_returns * 100, 
         label='Benchmark Daily Returns', linewidth=0.8, color='#A23B72', alpha=0.4)
ax2.set_title('Daily Returns', fontsize=12, fontweight='bold')
ax2.set_ylabel('Daily Return (%)', fontsize=10)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

# 3. Rolling Sharpe Ratio (20-day)
ax3 = axes[2]
rolling_window = 20
portfolio_rolling_sharpe = (portfolio_returns_series.rolling(rolling_window).mean() / 
                           portfolio_returns_series.rolling(rolling_window).std()) * np.sqrt(252)
benchmark_rolling_sharpe = (benchmark_returns.rolling(rolling_window).mean() / 
                           benchmark_returns.rolling(rolling_window).std()) * np.sqrt(252)

ax3.plot(portfolio_rolling_sharpe.index, portfolio_rolling_sharpe, 
         label='Portfolio Rolling Sharpe', linewidth=1.5, color='#2E86AB')
ax3.plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe, 
         label='Benchmark Rolling Sharpe', linewidth=1.5, color='#A23B72', alpha=0.7)
ax3.set_title(f'{rolling_window}-Day Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
ax3.set_ylabel('Sharpe Ratio', fontsize=10)
ax3.set_xlabel('Date', fontsize=10)
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('portfolio_returns.png', dpi=300, bbox_inches='tight')
print("\nSaved: portfolio_returns.png")

# Create a summary table
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

performance_data = pd.read_csv('performance_metrics.csv')
print("\n", performance_data.to_string(index=False))

print("\n" + "="*60)
print(f"Portfolio Composition:")
print("="*60)
print(portfolio.to_string(index=False))

print("\nAnalysis complete! Check 'portfolio_returns.png' for visualization.")

plt.show()
