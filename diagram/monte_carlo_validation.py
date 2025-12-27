import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAGRAM_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n" + "="*80)
print(" MONTE CARLO VALIDATION ")
print("="*80)

print("\nLoading data...")
portfolio_history = pd.read_csv(os.path.join(BASE_DIR, 'portfolio_history.csv'))
trades_log = pd.read_csv(os.path.join(BASE_DIR, 'trades_log.csv'))
performance_metrics = pd.read_csv(os.path.join(BASE_DIR, 'performance_metrics.csv'))

portfolio_history['Date'] = pd.to_datetime(portfolio_history['Date'])
portfolio_history.set_index('Date', inplace=True)
portfolio_returns = portfolio_history['Portfolio_Return'].dropna()

actual_cumulative_return = performance_metrics[performance_metrics['Metric'] == 'Cumulative_Return']['Portfolio'].values[0]
actual_sharpe = performance_metrics[performance_metrics['Metric'] == 'Sharpe_Ratio']['Portfolio'].values[0]
actual_sortino = performance_metrics[performance_metrics['Metric'] == 'Sortino_Ratio']['Portfolio'].values[0]
actual_max_drawdown = performance_metrics[performance_metrics['Metric'] == 'Max_Drawdown']['Portfolio'].values[0]

print(f"\nActual Model Performance:")
print(f"  Cumulative Return: {actual_cumulative_return:.4f} ({actual_cumulative_return*100:.2f}%)")
print(f"  Sharpe Ratio: {actual_sharpe:.4f}")
print(f"  Sortino Ratio: {actual_sortino:.4f}")
print(f"  Max Drawdown: {actual_max_drawdown:.4f} ({actual_max_drawdown*100:.2f}%)")

# Calculate actual trade statistics
if len(trades_log) > 0:
    sells = trades_log[trades_log['Action'] == 'SELL']
    if len(sells) > 0:
        actual_win_rate = len(sells[sells['Return'] > 0]) / len(sells)
        actual_avg_return = sells['Return'].mean()
        actual_num_trades = len(trades_log)
    else:
        actual_win_rate = 0
        actual_avg_return = 0
        actual_num_trades = len(trades_log)
else:
    actual_win_rate = 0
    actual_avg_return = 0
    actual_num_trades = 0

print(f"  Win Rate: {actual_win_rate:.2%}")
print(f"  Avg Trade Return: {actual_avg_return:.2%}")
print(f"  Total Trades: {actual_num_trades}")

print("\n" + "="*80)
print(" RUNNING MONTE CARLO SIMULATIONS ")
print("="*80)

N_SIMULATIONS = 10000
print(f"\nRunning {N_SIMULATIONS} simulations...")

import yfinance as yf
spy_data = yf.Ticker('SPY').history(start='2025-01-01', end='2025-12-24')
spy_returns = spy_data['Close'].pct_change().dropna()

market_mean = spy_returns.mean()
market_std = spy_returns.std()
n_periods = len(portfolio_returns)

mc_cumulative_returns = []
mc_sharpe_ratios = []
mc_sortino_ratios = []
mc_max_drawdowns = []

for i in range(N_SIMULATIONS):
    stock_specific_noise = np.random.normal(0, market_std * 0.5, n_periods)
    random_returns = np.random.normal(market_mean, market_std, n_periods) + stock_specific_noise
    
    trading_friction = np.random.uniform(-0.001, 0.001, n_periods)
    random_returns = random_returns + trading_friction
    
    # Cumulative return
    cum_ret = (1 + random_returns).prod() - 1
    mc_cumulative_returns.append(cum_ret)
    
    # Sharpe ratio
    sharpe = (random_returns.mean() * 252) / (random_returns.std() * np.sqrt(252)) if random_returns.std() > 0 else 0
    mc_sharpe_ratios.append(sharpe)
    
    # Sortino ratio
    downside = random_returns[random_returns < 0]
    sortino = (random_returns.mean() / np.sqrt((downside**2).mean()) * np.sqrt(252)) if len(downside) > 0 else 0
    mc_sortino_ratios.append(sortino)
    
    # Max drawdown
    cum = (1 + pd.Series(random_returns)).cumprod()
    running_max = cum.expanding().max()
    drawdown = ((cum - running_max) / running_max).min()
    mc_max_drawdowns.append(drawdown)
    
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{N_SIMULATIONS}")

print(f"\nDone")

mc_cumulative_returns = np.array(mc_cumulative_returns)
mc_sharpe_ratios = np.array(mc_sharpe_ratios)
mc_sortino_ratios = np.array(mc_sortino_ratios)
mc_max_drawdowns = np.array(mc_max_drawdowns)

print("\n" + "="*80)
print(" STATISTICAL TESTS ")
print("="*80)

print("\n1. CUMULATIVE RETURN TEST")
print("-" * 50)
percentile_cum = stats.percentileofscore(mc_cumulative_returns, actual_cumulative_return)
p_value_cum = (N_SIMULATIONS - percentile_cum * N_SIMULATIONS / 100) / N_SIMULATIONS
print(f"  Actual vs Monte Carlo:")
print(f"    Actual: {actual_cumulative_return:.4f}")
print(f"    MC Mean: {mc_cumulative_returns.mean():.4f}")
print(f"    MC Std: {mc_cumulative_returns.std():.4f}")
print(f"    Percentile: {percentile_cum:.2f}%")
print(f"    P-value: {p_value_cum:.4f}")
if p_value_cum < 0.05:
    print(f"    PASS: p < 0.05")
else:
    print(f"    FAIL: p >= 0.05")

print("\n2. SHARPE RATIO TEST")
print("-" * 50)
percentile_sharpe = stats.percentileofscore(mc_sharpe_ratios, actual_sharpe)
p_value_sharpe = (N_SIMULATIONS - percentile_sharpe * N_SIMULATIONS / 100) / N_SIMULATIONS
print(f"  Actual vs Monte Carlo:")
print(f"    Actual: {actual_sharpe:.4f}")
print(f"    MC Mean: {mc_sharpe_ratios.mean():.4f}")
print(f"    MC Std: {mc_sharpe_ratios.std():.4f}")
print(f"    Percentile: {percentile_sharpe:.2f}%")
print(f"    P-value: {p_value_sharpe:.4f}")
if p_value_sharpe < 0.05:
    print(f"    PASS: p < 0.05")
else:
    print(f"    FAIL: p >= 0.05")

print("\n3. SORTINO RATIO TEST")
print("-" * 50)
percentile_sortino = stats.percentileofscore(mc_sortino_ratios, actual_sortino)
p_value_sortino = (N_SIMULATIONS - percentile_sortino * N_SIMULATIONS / 100) / N_SIMULATIONS
print(f"  Actual vs Monte Carlo:")
print(f"    Actual: {actual_sortino:.4f}")
print(f"    MC Mean: {mc_sortino_ratios.mean():.4f}")
print(f"    MC Std: {mc_sortino_ratios.std():.4f}")
print(f"    Percentile: {percentile_sortino:.2f}%")
print(f"    P-value: {p_value_sortino:.4f}")
if p_value_sortino < 0.05:
    print(f"    PASS: p < 0.05")
else:
    print(f"    FAIL: p >= 0.05")

print("\n4. MAXIMUM DRAWDOWN TEST")
print("-" * 50)
percentile_dd = stats.percentileofscore(mc_max_drawdowns, actual_max_drawdown)
p_value_dd = percentile_dd / 100  # Lower is better for drawdown
print(f"  Actual vs Monte Carlo:")
print(f"    Actual: {actual_max_drawdown:.4f}")
print(f"    MC Mean: {mc_max_drawdowns.mean():.4f}")
print(f"    MC Std: {mc_max_drawdowns.std():.4f}")
print(f"    Percentile: {percentile_dd:.2f}%")
print(f"    P-value: {p_value_dd:.4f}")
if p_value_dd < 0.05:
    print(f"    PASS: p < 0.05")
else:
    print(f"    FAIL: p >= 0.05")

print("\n5. NORMALITY TEST (Jarque-Bera)")
print("-" * 50)
jb_stat, jb_pvalue = stats.jarque_bera(portfolio_returns)
print(f"  Test Statistic: {jb_stat:.4f}")
print(f"  P-value: {jb_pvalue:.4f}")
if jb_pvalue < 0.05:
    print(f"    Non-normal distribution (fat tails)")
else:
    print(f"    Approximately normal")

print("\n6. AUTOCORRELATION TEST (Ljung-Box)")
print("-" * 50)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_result = acorr_ljungbox(portfolio_returns.dropna(), lags=[10], return_df=True)
lb_stat = lb_result['lb_stat'].values[0]
lb_pvalue = lb_result['lb_pvalue'].values[0]
print(f"  Test Statistic: {lb_stat:.4f}")
print(f"  P-value: {lb_pvalue:.4f}")
if lb_pvalue < 0.05:
    print(f"    PASS: Autocorrelation detected")
else:
    print(f"    No autocorrelation")

print("\n7. WIN RATE SIGNIFICANCE TEST")
print("-" * 50)
if actual_num_trades > 0 and len(sells) > 0:
    n_trades = len(sells)
    n_wins = len(sells[sells['Return'] > 0])
    binom_result = stats.binomtest(n_wins, n_trades, 0.5, alternative='greater')
    binom_pvalue = binom_result.pvalue
    print(f"  Win Rate: {actual_win_rate:.2%}")
    print(f"  Trades: {n_wins}/{n_trades}")
    print(f"  P-value (vs 50%): {binom_pvalue:.4f}")
    if binom_pvalue < 0.05:
        print(f"    PASS: Win rate > 50%")
    else:
        print(f"    FAIL: Win rate not significant")
else:
    print(f"  Not enough trades to test")

print("\n8. MEAN REVERSION TEST (Augmented Dickey-Fuller)")
print("-" * 50)
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(portfolio_returns.dropna())
adf_stat = adf_result[0]
adf_pvalue = adf_result[1]
print(f"  ADF Statistic: {adf_stat:.4f}")
print(f"  P-value: {adf_pvalue:.4f}")
if adf_pvalue < 0.05:
    print(f"    PASS: Stationary")
else:
    print(f"    Non-stationary")

print("\n9. TRADE-LEVEL BOOTSTRAP TEST")
print("-" * 50)
if len(sells) > 0:
    actual_trade_returns = sells['Return'].values
    n_bootstrap = 10000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(actual_trade_returns, size=len(actual_trade_returns), replace=True)
        bootstrap_means.append(sample.mean())
    
    bootstrap_means = np.array(bootstrap_means)
    ci_lower_boot = np.percentile(bootstrap_means, 2.5)
    ci_upper_boot = np.percentile(bootstrap_means, 97.5)
    
    print(f"  Actual Avg Trade Return: {actual_avg_return:.2%}")
    print(f"  Bootstrap 95% CI: [{ci_lower_boot:.2%}, {ci_upper_boot:.2%}]")
    
    if ci_lower_boot > 0:
        print(f"    PASS: CI > 0")
        bootstrap_significant = True
    else:
        print(f"    FAIL: CI includes zero")
        bootstrap_significant = False
else:
    bootstrap_significant = False
    print(f"  Not enough trades to test")

print("\n10. INFORMATION RATIO TEST")
print("-" * 50)
spy_returns_daily = spy_data['Close'].pct_change().dropna()
common_dates = portfolio_returns.index.intersection(spy_returns_daily.index)
if len(common_dates) > 20:
    port_aligned = portfolio_returns.loc[common_dates]
    spy_aligned = spy_returns_daily.loc[common_dates]
    excess_returns = port_aligned - spy_aligned
    tracking_error = excess_returns.std() * np.sqrt(252)
    info_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
    
    print(f"  Information Ratio: {info_ratio:.4f}")
    print(f"  Tracking Error: {tracking_error:.4f}")
    
    if info_ratio > 0.5:
        print(f"    PASS: IR > 0.5")
        ir_significant = True
    else:
        print(f"    FAIL: IR < 0.5")
        ir_significant = False
else:
    ir_significant = False
    info_ratio = 0
    print(f"  Not enough data to calculate")

# Overall Assessment
print("\n" + "="*80)
print(" OVERALL MODEL VALIDATION ")
print("="*80)

tests_passed = 0
total_tests = 0

test_results = [
    ("Cumulative Return vs Random", p_value_cum < 0.05),
    ("Sharpe Ratio vs Random", p_value_sharpe < 0.05),
    ("Sortino Ratio vs Random", p_value_sortino < 0.05),
    ("Max Drawdown Control", p_value_dd < 0.05),
    ("Stationarity (MR Valid)", adf_pvalue < 0.05),
    ("Trade Bootstrap CI > 0", bootstrap_significant if len(sells) > 0 else False),
    ("Information Ratio > 0.5", ir_significant)
]

if actual_num_trades > 0 and len(sells) > 0:
    test_results.append(("Win Rate", binom_pvalue < 0.05))

print("\nResults:")
for test_name, passed in test_results:
    total_tests += 1
    if passed:
        tests_passed += 1
        print(f"  [PASS] {test_name}")
    else:
        print(f"  [FAIL] {test_name}")

pass_rate = tests_passed / total_tests
print(f"\nTests Passed: {tests_passed}/{total_tests} ({pass_rate:.1%})")

if pass_rate >= 0.7:
    print("\nVALIDATION: STRONG")
elif pass_rate >= 0.5:
    print("\nVALIDATION: MODERATE")
else:
    print("\nVALIDATION: WEAK")

print("\n" + "="*80)
print(" GENERATING VISUALIZATIONS ")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Monte Carlo Simulation Results vs Actual Model', fontsize=16, fontweight='bold')

# 1. Cumulative Returns Distribution
ax1 = axes[0, 0]
ax1.hist(mc_cumulative_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(actual_cumulative_return, color='red', linestyle='--', linewidth=2, label=f'Actual: {actual_cumulative_return:.2%}')
ax1.axvline(mc_cumulative_returns.mean(), color='green', linestyle='--', linewidth=2, label=f'MC Mean: {mc_cumulative_returns.mean():.2%}')
ax1.set_xlabel('Cumulative Return', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Cumulative Returns Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add percentile text
percentile_text = f'Actual at {percentile_cum:.1f}th percentile'
ax1.text(0.05, 0.95, percentile_text, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Sharpe Ratio Distribution
ax2 = axes[0, 1]
ax2.hist(mc_sharpe_ratios, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
ax2.axvline(actual_sharpe, color='red', linestyle='--', linewidth=2, label=f'Actual: {actual_sharpe:.3f}')
ax2.axvline(mc_sharpe_ratios.mean(), color='green', linestyle='--', linewidth=2, label=f'MC Mean: {mc_sharpe_ratios.mean():.3f}')
ax2.set_xlabel('Sharpe Ratio', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Sharpe Ratio Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

percentile_text = f'Actual at {percentile_sharpe:.1f}th percentile'
ax2.text(0.05, 0.95, percentile_text, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Maximum Drawdown Distribution
ax3 = axes[1, 0]
ax3.hist(mc_max_drawdowns, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
ax3.axvline(actual_max_drawdown, color='red', linestyle='--', linewidth=2, label=f'Actual: {actual_max_drawdown:.2%}')
ax3.axvline(mc_max_drawdowns.mean(), color='green', linestyle='--', linewidth=2, label=f'MC Mean: {mc_max_drawdowns.mean():.2%}')
ax3.set_xlabel('Maximum Drawdown', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Maximum Drawdown Distribution', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

percentile_text = f'Actual at {percentile_dd:.1f}th percentile\n(Lower is better)'
ax3.text(0.05, 0.95, percentile_text, transform=ax3.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Sortino Ratio Distribution
ax4 = axes[1, 1]
ax4.hist(mc_sortino_ratios, bins=50, alpha=0.7, color='plum', edgecolor='black')
ax4.axvline(actual_sortino, color='red', linestyle='--', linewidth=2, label=f'Actual: {actual_sortino:.3f}')
ax4.axvline(mc_sortino_ratios.mean(), color='green', linestyle='--', linewidth=2, label=f'MC Mean: {mc_sortino_ratios.mean():.3f}')
ax4.set_xlabel('Sortino Ratio', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Sortino Ratio Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

percentile_text = f'Actual at {percentile_sortino:.1f}th percentile'
ax4.text(0.05, 0.95, percentile_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(DIAGRAM_DIR, 'monte_carlo_validation.png'), dpi=300, bbox_inches='tight')
print("\nSaved: monte_carlo_validation.png")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Return Distribution Analysis', fontsize=16, fontweight='bold')

# Q-Q Plot
ax_qq = axes2[0]
stats.probplot(portfolio_returns.dropna(), dist="norm", plot=ax_qq)
ax_qq.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
ax_qq.grid(True, alpha=0.3)

# Distribution histogram 
ax_dist = axes2[1]
ax_dist.hist(portfolio_returns.dropna(), bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
ax_dist.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
ax_dist.set_xlabel('Daily Returns', fontsize=11)
ax_dist.set_ylabel('Density', fontsize=11)
ax_dist.set_title('Returns Distribution vs Normal', fontsize=12, fontweight='bold')
ax_dist.legend()
ax_dist.grid(True, alpha=0.3)

# text
skew = portfolio_returns.skew()
kurt = portfolio_returns.kurtosis()
stats_text = f'Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}'
ax_dist.text(0.05, 0.95, stats_text, transform=ax_dist.transAxes, 
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(DIAGRAM_DIR, 'return_distribution_analysis.png'), dpi=300, bbox_inches='tight')
print("Saved: return_distribution_analysis.png")

# Save summary statistics
summary_stats = pd.DataFrame({
    'Metric': ['Cumulative Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
    'Actual': [actual_cumulative_return, actual_sharpe, actual_sortino, actual_max_drawdown],
    'MC_Mean': [mc_cumulative_returns.mean(), mc_sharpe_ratios.mean(), mc_sortino_ratios.mean(), mc_max_drawdowns.mean()],
    'MC_Std': [mc_cumulative_returns.std(), mc_sharpe_ratios.std(), mc_sortino_ratios.std(), mc_max_drawdowns.std()],
    'Percentile': [percentile_cum, percentile_sharpe, percentile_sortino, percentile_dd],
    'P_Value': [p_value_cum, p_value_sharpe, p_value_sortino, p_value_dd],
    'Significant': [p_value_cum < 0.05, p_value_sharpe < 0.05, p_value_sortino < 0.05, p_value_dd < 0.05]
})

summary_stats.to_csv(os.path.join(BASE_DIR, 'monte_carlo_summary.csv'), index=False)
print("Saved: monte_carlo_summary.csv")

print("\nGenerating paths comparison...")

fig3, axes3 = plt.subplots(2, 2, figsize=(18, 14))
fig3.suptitle('Model Equity Path vs Monte Carlo Simulations', fontsize=16, fontweight='bold')

# Calculate actual model equity curve
actual_equity = (1 + portfolio_returns).cumprod()
dates = actual_equity.index

# Generate MC equity paths for visualization
N_PATHS_DISPLAY = min(100, N_SIMULATIONS)
mc_equity_paths = []

np.random.seed(42)
for i in range(N_PATHS_DISPLAY):
    stock_specific_noise = np.random.normal(0, market_std * 0.5, n_periods)
    random_returns = np.random.normal(market_mean, market_std, n_periods) + stock_specific_noise
    trading_friction = np.random.uniform(-0.001, 0.001, n_periods)
    random_returns = random_returns + trading_friction
    equity_path = (1 + pd.Series(random_returns, index=dates)).cumprod()
    mc_equity_paths.append(equity_path)

# 1. All MC paths with model overlay
ax1 = axes3[0, 0]
for i, path in enumerate(mc_equity_paths):
    ax1.plot(path.index, path.values, alpha=0.15, color='gray', linewidth=0.8)
ax1.plot(actual_equity.index, actual_equity.values, color='red', linewidth=2.5, label='Model', zorder=10)

# Add percentile bands
mc_equity_array = np.array([path.values for path in mc_equity_paths])
percentile_5 = np.percentile(mc_equity_array, 5, axis=0)
percentile_25 = np.percentile(mc_equity_array, 25, axis=0)
percentile_50 = np.percentile(mc_equity_array, 50, axis=0)
percentile_75 = np.percentile(mc_equity_array, 75, axis=0)
percentile_95 = np.percentile(mc_equity_array, 95, axis=0)

ax1.fill_between(dates, percentile_5, percentile_95, alpha=0.2, color='blue', label='5th-95th Percentile')
ax1.fill_between(dates, percentile_25, percentile_75, alpha=0.3, color='blue', label='25th-75th Percentile')
ax1.plot(dates, percentile_50, color='blue', linestyle='--', linewidth=1.5, label='MC Median')

ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Portfolio Value (Starting = 1.0)', fontsize=11)
ax1.set_title(f'Model Path vs {N_PATHS_DISPLAY} Monte Carlo Paths', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Final value distribution with model position
ax2 = axes3[0, 1]
final_values = mc_equity_array[:, -1]
actual_final = actual_equity.iloc[-1]

ax2.hist(final_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
ax2.axvline(actual_final, color='red', linestyle='--', linewidth=2.5, label=f'Model: {actual_final:.3f}')
ax2.axvline(np.median(final_values), color='green', linestyle='--', linewidth=2, label=f'MC Median: {np.median(final_values):.3f}')

# Add kernel density estimate
from scipy.stats import gaussian_kde
kde = gaussian_kde(final_values)
x_kde = np.linspace(final_values.min(), final_values.max(), 200)
ax2.plot(x_kde, kde(x_kde), color='darkblue', linewidth=2, label='KDE')

ax2.set_xlabel('Final Portfolio Value', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Distribution of Final Portfolio Values', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add percentile annotation
model_percentile = stats.percentileofscore(final_values, actual_final)
ax2.text(0.05, 0.95, f'Model at {model_percentile:.1f}th percentile', transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 3. Excess return over time (Model - MC Median)
ax3 = axes3[1, 0]
excess_over_mc = actual_equity.values - percentile_50
ax3.fill_between(dates, 0, excess_over_mc, where=(excess_over_mc >= 0), 
                  interpolate=True, alpha=0.6, color='green', label='Outperformance')
ax3.fill_between(dates, 0, excess_over_mc, where=(excess_over_mc < 0), 
                  interpolate=True, alpha=0.6, color='red', label='Underperformance')
ax3.axhline(0, color='black', linewidth=1)
ax3.plot(dates, excess_over_mc, color='black', linewidth=1.5)

ax3.set_xlabel('Date', fontsize=11)
ax3.set_ylabel('Excess Return vs MC Median', fontsize=11)
ax3.set_title('Model Excess Return Over Random Selection', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Add summary stats
total_outperform_days = np.sum(excess_over_mc > 0)
total_days = len(excess_over_mc)
avg_excess = np.mean(excess_over_mc)
ax3.text(0.95, 0.95, f'Outperform Days: {total_outperform_days}/{total_days} ({total_outperform_days/total_days*100:.1f}%)\nAvg Excess: {avg_excess:.4f}', 
         transform=ax3.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 4. Rolling percentile of model vs MC
ax4 = axes3[1, 1]
rolling_percentiles = []
window_size = 10  # 10-day rolling window

for i in range(len(dates)):
    if i < window_size:
        # For initial period, use available data
        mc_vals = mc_equity_array[:, :i+1].mean(axis=1) if i > 0 else mc_equity_array[:, 0]
        model_val = actual_equity.iloc[:i+1].mean() if i > 0 else actual_equity.iloc[0]
    else:
        # Use rolling window
        mc_vals = mc_equity_array[:, i-window_size:i+1].mean(axis=1)
        model_val = actual_equity.iloc[i-window_size:i+1].mean()
    
    pct = stats.percentileofscore(mc_vals, model_val)
    rolling_percentiles.append(pct)

rolling_percentiles = np.array(rolling_percentiles)

# Color based on percentile
colors = np.where(rolling_percentiles >= 50, 'green', 'red')
ax4.scatter(dates, rolling_percentiles, c=colors, s=10, alpha=0.6)
ax4.plot(dates, rolling_percentiles, color='black', linewidth=1, alpha=0.5)

ax4.axhline(50, color='gray', linestyle='--', linewidth=1.5, label='50th Percentile')
ax4.axhline(75, color='green', linestyle=':', linewidth=1, alpha=0.7, label='75th Percentile')
ax4.axhline(25, color='red', linestyle=':', linewidth=1, alpha=0.7, label='25th Percentile')

ax4.fill_between(dates, 50, rolling_percentiles, where=(rolling_percentiles >= 50),
                  interpolate=True, alpha=0.3, color='green')
ax4.fill_between(dates, 50, rolling_percentiles, where=(rolling_percentiles < 50),
                  interpolate=True, alpha=0.3, color='red')

ax4.set_xlabel('Date', fontsize=11)
ax4.set_ylabel('Model Percentile vs MC', fontsize=11)
ax4.set_title('Rolling Model Rank Among MC Simulations', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 100)
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# Add summary
avg_percentile = rolling_percentiles.mean()
above_50_pct = np.sum(rolling_percentiles > 50) / len(rolling_percentiles) * 100
ax4.text(0.05, 0.05, f'Avg Percentile: {avg_percentile:.1f}\nAbove Median: {above_50_pct:.1f}% of time', 
         transform=ax4.transAxes, fontsize=10, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(DIAGRAM_DIR, 'monte_carlo_paths_comparison.png'), dpi=300, bbox_inches='tight')
print("Saved: monte_carlo_paths_comparison.png")

print("\n" + "="*80)
print(" COMPLETE ")
print("="*80)
print()
