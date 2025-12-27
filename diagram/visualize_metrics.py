import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load all data
performance = pd.read_csv('performance_metrics.csv')
portfolio_history = pd.read_csv('portfolio_history.csv', parse_dates=['Date'], index_col='Date')
trades_log = pd.read_csv('trades_log.csv', parse_dates=['Date'])
portfolio_comp = pd.read_csv('portfolio_composition.csv')

print(f"Portfolio history: {len(portfolio_history)} days")
print(f"Trades executed: {len(trades_log)}")
print(f"Current positions: {len(portfolio_comp)}")

# Calculate returns
portfolio_history['Portfolio_Return_Pct'] = (portfolio_history['Portfolio_Value'] / 100000 - 1) * 100
portfolio_history['Daily_Return'] = portfolio_history['Portfolio_Value'].pct_change() * 100

# Calculate benchmark returns (SPY normalized to same start)
spy_start_price = portfolio_history['SPY_Price'].iloc[0]
portfolio_history['SPY_Return_Pct'] = (portfolio_history['SPY_Price'] / spy_start_price - 1) * 100

# Calculate drawdowns
portfolio_history['Portfolio_Peak'] = portfolio_history['Portfolio_Value'].expanding().max()
portfolio_history['Portfolio_Drawdown'] = ((portfolio_history['Portfolio_Value'] - portfolio_history['Portfolio_Peak']) / portfolio_history['Portfolio_Peak']) * 100

portfolio_history['SPY_Peak'] = portfolio_history['SPY_Price'].expanding().max()
portfolio_history['SPY_Drawdown'] = ((portfolio_history['SPY_Price'] - portfolio_history['SPY_Peak']) / portfolio_history['SPY_Peak']) * 100

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

fig.suptitle('Performance Analysis', 
             fontsize=18, fontweight='bold', y=0.995)

#  Cumulative Returns
ax1 = fig.add_subplot(gs[0, :])

ax1.plot(portfolio_history.index, portfolio_history['Portfolio_Return_Pct'], 
         label='Dynamic Portfolio', linewidth=2.5, color='#2E86AB', zorder=3)
ax1.plot(portfolio_history.index, portfolio_history['SPY_Return_Pct'], 
         label='S&P 500 Benchmark', linewidth=2, color='#A23B72', alpha=0.7, zorder=2)

# Mark buy and sell trades
buys = trades_log[trades_log['Action'] == 'BUY']
sells = trades_log[trades_log['Action'] == 'SELL']

for _, trade in sells.iterrows():
    if trade['Date'] in portfolio_history.index:
        y_val = portfolio_history.loc[trade['Date'], 'Portfolio_Return_Pct']
        color = '#00C853' if trade['Return'] > 0 else '#D50000'
        marker = '^' if trade['Reason'] == 'TAKE_PROFIT' else 'v' if trade['Reason'] == 'STOP_LOSS' else 'o'
        ax1.scatter(trade['Date'], y_val, color=color, marker=marker, s=30, alpha=0.6, zorder=4) # pyright: ignore[reportArgumentType]

ax1.set_title('Cumulative Returns Over Time with Trade Markers', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

# Add performance box
final_port = portfolio_history['Portfolio_Return_Pct'].iloc[-1]
final_bench = portfolio_history['SPY_Return_Pct'].iloc[-1]
outperform = final_port - final_bench

textstr = f'Final Returns:\nPortfolio: {final_port:.2f}%\nBenchmark: {final_bench:.2f}%\nOutperformance: {outperform:.2f}%'
ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# drawdown analysis
ax2 = fig.add_subplot(gs[1, :])

ax2.fill_between(portfolio_history.index, portfolio_history['Portfolio_Drawdown'], 0, 
                  color='#2E86AB', alpha=0.3, label='Portfolio Drawdown')
ax2.fill_between(portfolio_history.index, portfolio_history['SPY_Drawdown'], 0, 
                  color='#A23B72', alpha=0.2, label='Benchmark Drawdown')
ax2.plot(portfolio_history.index, portfolio_history['Portfolio_Drawdown'], 
         color='#2E86AB', linewidth=1.5)
ax2.plot(portfolio_history.index, portfolio_history['SPY_Drawdown'], 
         color='#A23B72', linewidth=1.5, alpha=0.7)

ax2.set_title('Drawdown Analysis', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# Add max drawdown info
port_max_dd = portfolio_history['Portfolio_Drawdown'].min()
bench_max_dd = portfolio_history['SPY_Drawdown'].min()
textstr = f'Max Drawdown:\nPortfolio: {port_max_dd:.2f}%\nBenchmark: {bench_max_dd:.2f}%'
ax2.text(0.98, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# number of active positions
ax3 = fig.add_subplot(gs[2, 0])

ax3.plot(portfolio_history.index, portfolio_history['Positions'], 
         color='#F77F00', linewidth=2, marker='o', markersize=2)
ax3.fill_between(portfolio_history.index, portfolio_history['Positions'], 
                  alpha=0.3, color='#F77F00')

ax3.set_title('Active Positions Over Time', fontsize=12, fontweight='bold', pad=10)
ax3.set_ylabel('Number of Stocks', fontsize=10)
ax3.set_ylim(0, 8)
ax3.grid(True, alpha=0.3, linestyle='--')

avg_positions = portfolio_history['Positions'].mean()
ax3.axhline(y=avg_positions, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Avg: {avg_positions:.1f}')
ax3.legend(loc='best', fontsize=9)

# trading activity
ax4 = fig.add_subplot(gs[2, 1])

# Count trades by reason
sell_reasons = sells.groupby('Reason').size()
buy_reasons = buys.groupby('Reason').size()

trade_data = pd.DataFrame({
    'Buys': buy_reasons,
    'Sells': sell_reasons
}).fillna(0)

x = np.arange(len(trade_data))
width = 0.35

bars1 = ax4.bar(x - width/2, trade_data['Buys'], width, label='Buys', color='#00C853', alpha=0.8)
bars2 = ax4.bar(x + width/2, trade_data['Sells'], width, label='Sells', color='#D50000', alpha=0.8)

ax4.set_title('Trading Activity by Type', fontsize=12, fontweight='bold', pad=10)
ax4.set_ylabel('Number of Trades', fontsize=10)
ax4.set_xticks(x)
ax4.set_xticklabels(trade_data.index, rotation=45, ha='right', fontsize=9)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

# performance metrics comparison
ax5 = fig.add_subplot(gs[3, 0])

metrics = performance['Metric'].tolist()
port_values = performance['Portfolio'].tolist()
bench_values = performance['Benchmark'].tolist()

x = np.arange(len(metrics))
width = 0.35

bars1 = ax5.barh(x - width/2, port_values, width, label='Portfolio', color='#2E86AB', alpha=0.8)
bars2 = ax5.barh(x + width/2, bench_values, width, label='Benchmark', color='#A23B72', alpha=0.8)

ax5.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold', pad=10)
ax5.set_yticks(x)
ax5.set_yticklabels(metrics, fontsize=9)
ax5.set_xlabel('Value', fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='x', linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        width_val = bar.get_width()
        ax5.text(width_val, bar.get_y() + bar.get_height()/2.,
                f'{width_val:.3f}', ha='left', va='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

#portfolio composition
ax6 = fig.add_subplot(gs[3, 1])

colors_pie = plt.cm.Set3(np.linspace(0, 1, len(portfolio_comp)))
wedges, texts, autotexts = ax6.pie(portfolio_comp['Weight'], 
                                     labels=portfolio_comp['Ticker'],
                                     autopct='%1.1f%%',
                                     colors=colors_pie,
                                     startangle=90,
                                     textprops={'fontsize': 9})

ax6.set_title('Final Portfolio Composition', fontsize=12, fontweight='bold', pad=10)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)

#summary statistics
summary_text = f"""
SUMMARY STATISTICS
{'='*50}
Period: {portfolio_history.index[0].strftime('%Y-%m-%d')} to {portfolio_history.index[-1].strftime('%Y-%m-%d')}
Trading Days: {len(portfolio_history)}
Total Trades: {len(trades_log)} ({len(buys)} buys, {len(sells)} sells)

Portfolio Performance:
  Final Value: ${portfolio_history['Portfolio_Value'].iloc[-1]:,.0f}
  Total Return: {final_port:.2f}%
  Sortino Ratio: {performance.loc[performance['Metric']=='Sortino_Ratio', 'Portfolio'].values[0]:.2f}
  Sharpe Ratio: {performance.loc[performance['Metric']=='Sharpe_Ratio', 'Portfolio'].values[0]:.2f}
  Max Drawdown: {port_max_dd:.2f}%
  Volatility: {performance.loc[performance['Metric']=='Volatility', 'Portfolio'].values[0]:.2f}

Win Rate: {len(sells[sells['Return'] > 0]) / len(sells) * 100:.1f}% ({len(sells[sells['Return'] > 0])} of {len(sells)} sells)
Avg Trade Return: {sells['Return'].mean() * 100:.2f}%
"""

plt.figtext(0.02, 0.02, summary_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n file Saved")

# Create a second figure for detailed trade analysis
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle('Detailed Trade Analysis', fontsize=16, fontweight='bold')

# Trade returns distribution
ax_tr = axes[0, 0]
sell_returns = sells['Return'] * 100
ax_tr.hist(sell_returns, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
ax_tr.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
ax_tr.axvline(x=sell_returns.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {sell_returns.mean():.2f}%')
ax_tr.set_title('Distribution of Trade Returns', fontsize=12, fontweight='bold')
ax_tr.set_xlabel('Return (%)', fontsize=10)
ax_tr.set_ylabel('Frequency', fontsize=10)
ax_tr.legend()
ax_tr.grid(True, alpha=0.3)

# Trade returns by reason
ax_rr = axes[0, 1]
reason_returns = sells.groupby('Reason')['Return'].apply(list)
box_data = [reason_returns[reason] for reason in reason_returns.index]
bp = ax_rr.boxplot([np.array(d)*100 for d in box_data], labels=reason_returns.index, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#F77F00')
    patch.set_alpha(0.7)
ax_rr.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax_rr.set_title('Returns by Sell Reason', fontsize=12, fontweight='bold')
ax_rr.set_ylabel('Return (%)', fontsize=10)
ax_rr.grid(True, alpha=0.3, axis='y')
ax_rr.tick_params(axis='x', rotation=45)

# Most traded stocks
ax_ts = axes[1, 0]
all_tickers = pd.concat([buys['Ticker'], sells['Ticker']])
top_traded = all_tickers.value_counts().head(10)
ax_ts.barh(range(len(top_traded)), top_traded.values, color='#2E86AB', alpha=0.8)
ax_ts.set_yticks(range(len(top_traded)))
ax_ts.set_yticklabels(top_traded.index)
ax_ts.set_title('Top 10 Most Traded Stocks', fontsize=12, fontweight='bold')
ax_ts.set_xlabel('Number of Trades', fontsize=10)
ax_ts.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(top_traded.values):
    ax_ts.text(v, i, f' {v}', va='center', fontsize=9)

# Trade value distribution
ax_tv = axes[1, 1]
trade_values = trades_log['Value'] / 1000  # Convert to thousands
ax_tv.scatter(range(len(trade_values)), trade_values, 
             c=['#00C853' if action == 'BUY' else '#D50000' for action in trades_log['Action']],
             alpha=0.5, s=20)
ax_tv.set_title('Trade Values Over Time', fontsize=12, fontweight='bold')
ax_tv.set_xlabel('Trade Number', fontsize=10)
ax_tv.set_ylabel('Trade Value ($K)', fontsize=10)
ax_tv.grid(True, alpha=0.3)

# Legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#00C853', alpha=0.7, label='Buy'),
                  Patch(facecolor='#D50000', alpha=0.7, label='Sell')]
ax_tv.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.savefig('trade_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: trade_analysis.png")

print("VISUALIZATION COMPLETE")
plt.show()
