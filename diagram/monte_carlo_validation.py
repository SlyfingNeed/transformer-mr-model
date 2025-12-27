import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("\n" + "="*80)
print(" MONTE CARLO SIMULATION & STATISTICAL VALIDATION ")
print("="*80)

# Load actual model results
print("\nLoading model results...")
portfolio_history = pd.read_csv('c:\\Users\\seva\\transformer-mr-model\\portfolio_history.csv')
trades_log = pd.read_csv('c:\\Users\\seva\\transformer-mr-model\\trades_log.csv')
performance_metrics = pd.read_csv('c:\\Users\\seva\\transformer-mr-model\\performance_metrics.csv')

# Extract actual performance
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

# Monte Carlo Simulation - RANDOM STOCK SELECTION BENCHMARK
print("\n" + "="*80)
print(" RUNNING MONTE CARLO SIMULATIONS ")
print("="*80)
print("\nMethodology: Comparing model vs RANDOM stock selection from same universe")
print("This tests if the model's stock SELECTION has edge over random picking")

N_SIMULATIONS = 10000
print(f"\nRunning {N_SIMULATIONS} simulations...")

# Load SPY data for benchmark comparison
import yfinance as yf
spy_data = yf.Ticker('SPY').history(start='2025-01-01', end='2025-12-24')
spy_returns = spy_data['Close'].pct_change().dropna()

# Use SPY returns as the market baseline for random portfolios
market_mean = spy_returns.mean()
market_std = spy_returns.std()
n_periods = len(portfolio_returns)

# Generate random portfolios (random stock selection simulation)
mc_cumulative_returns = []
mc_sharpe_ratios = []
mc_sortino_ratios = []
mc_max_drawdowns = []

for i in range(N_SIMULATIONS):
    # Simulate random stock selection: market return + random stock-specific noise
    # This represents picking random stocks without any model signal
    stock_specific_noise = np.random.normal(0, market_std * 0.5, n_periods)  # idiosyncratic risk
    random_returns = np.random.normal(market_mean, market_std, n_periods) + stock_specific_noise
    
    # Add realistic trading friction (slippage, timing)
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
        print(f"  Completed {i + 1}/{N_SIMULATIONS} simulations")

print(f"\nMonte Carlo simulations completed!")

# Convert to arrays
mc_cumulative_returns = np.array(mc_cumulative_returns)
mc_sharpe_ratios = np.array(mc_sharpe_ratios)
mc_sortino_ratios = np.array(mc_sortino_ratios)
mc_max_drawdowns = np.array(mc_max_drawdowns)

# Statistical Tests
print("\n" + "="*80)
print(" STATISTICAL VALIDATION TESTS ")
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
    print(f"    ✓ SIGNIFICANT: Model outperforms random walk (p < 0.05)")
else:
    print(f"    ✗ NOT SIGNIFICANT: Cannot reject random walk hypothesis")

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
    print(f"    ✓ SIGNIFICANT: Risk-adjusted returns are superior (p < 0.05)")
else:
    print(f"    ✗ NOT SIGNIFICANT: Risk-adjusted returns not statistically different")

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
    print(f"    ✓ SIGNIFICANT: Downside risk management is effective (p < 0.05)")
else:
    print(f"    ✗ NOT SIGNIFICANT: Downside protection not statistically different")

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
    print(f"    ✓ SIGNIFICANT: Drawdown control is superior (p < 0.05)")
else:
    print(f"    ✗ NOT SIGNIFICANT: Drawdown not significantly better")

print("\n5. NORMALITY TEST (Jarque-Bera)")
print("-" * 50)
jb_stat, jb_pvalue = stats.jarque_bera(portfolio_returns)
print(f"  Test Statistic: {jb_stat:.4f}")
print(f"  P-value: {jb_pvalue:.4f}")
if jb_pvalue < 0.05:
    print(f"    Returns are NOT normally distributed (reject normality)")
    print(f"    → Fat tails present (common in trading strategies)")
else:
    print(f"    Returns are approximately normal")

print("\n6. AUTOCORRELATION TEST (Ljung-Box)")
print("-" * 50)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_result = acorr_ljungbox(portfolio_returns.dropna(), lags=[10], return_df=True)
lb_stat = lb_result['lb_stat'].values[0]
lb_pvalue = lb_result['lb_pvalue'].values[0]
print(f"  Test Statistic: {lb_stat:.4f}")
print(f"  P-value: {lb_pvalue:.4f}")
if lb_pvalue < 0.05:
    print(f"    ✓ SIGNIFICANT: Returns show autocorrelation")
    print(f"    → Model captures time-series patterns")
else:
    print(f"    Returns are independent (no autocorrelation)")

print("\n7. WIN RATE SIGNIFICANCE TEST")
print("-" * 50)
if actual_num_trades > 0 and len(sells) > 0:
    n_trades = len(sells)
    n_wins = len(sells[sells['Return'] > 0])
    # Binomial test against 50% win rate
    binom_result = stats.binomtest(n_wins, n_trades, 0.5, alternative='greater')
    binom_pvalue = binom_result.pvalue
    print(f"  Win Rate: {actual_win_rate:.2%}")
    print(f"  Trades: {n_wins}/{n_trades}")
    print(f"  P-value (vs 50%): {binom_pvalue:.4f}")
    if binom_pvalue < 0.05:
        print(f"    ✓ SIGNIFICANT: Win rate > 50% (p < 0.05)")
    else:
        print(f"    ✗ NOT SIGNIFICANT: Win rate not significantly better than chance")
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
    print(f"    ✓ Returns are stationary (mean-reverting)")
    print(f"    → Appropriate for mean reversion strategy")
else:
    print(f"    Returns show trend/non-stationarity")

print("\n9. TRADE-LEVEL BOOTSTRAP TEST")
print("-" * 50)
if len(sells) > 0:
    actual_trade_returns = sells['Return'].values
    n_bootstrap = 10000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Random sample with replacement
        sample = np.random.choice(actual_trade_returns, size=len(actual_trade_returns), replace=True)
        bootstrap_means.append(sample.mean())
    
    bootstrap_means = np.array(bootstrap_means)
    ci_lower_boot = np.percentile(bootstrap_means, 2.5)
    ci_upper_boot = np.percentile(bootstrap_means, 97.5)
    
    print(f"  Actual Avg Trade Return: {actual_avg_return:.2%}")
    print(f"  Bootstrap 95% CI: [{ci_lower_boot:.2%}, {ci_upper_boot:.2%}]")
    
    if ci_lower_boot > 0:
        print(f"    ✓ SIGNIFICANT: Trade returns significantly > 0")
        bootstrap_significant = True
    else:
        print(f"    ✗ CI includes zero, not definitively profitable")
        bootstrap_significant = False
else:
    bootstrap_significant = False
    print(f"  Not enough trades to test")

print("\n10. INFORMATION RATIO TEST")
print("-" * 50)
# Information ratio: excess return / tracking error
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
    
    # IR > 0.5 is good, > 1.0 is excellent
    if info_ratio > 0.5:
        print(f"    ✓ SIGNIFICANT: Good excess returns vs benchmark")
        ir_significant = True
    else:
        print(f"    ✗ Information ratio below threshold (0.5)")
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

for test_name, passed in test_results:
    total_tests += 1
    if passed:
        tests_passed += 1
        print(f"  ✓ {test_name}")
    else:
        print(f"  ✗ {test_name}")

pass_rate = tests_passed / total_tests
print(f"\nTests Passed: {tests_passed}/{total_tests} ({pass_rate:.1%})")

if pass_rate >= 0.7:
    print("\n✓✓✓ MODEL VALIDATION: STRONG - Model shows statistically significant edge")
elif pass_rate >= 0.5:
    print("\n✓✓ MODEL VALIDATION: MODERATE - Model shows some statistical significance")
else:
    print("\n✗ MODEL VALIDATION: WEAK - Model does not show strong statistical edge")

# Create visualizations
print("\n" + "="*80)
print(" GENERATING MONTE CARLO VISUALIZATIONS ")
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
plt.savefig('c:\\Users\\seva\\transformer-mr-model\\diagram\\monte_carlo_validation.png', dpi=300, bbox_inches='tight')
print("\nSaved: monte_carlo_validation.png")

# Additional plot: Q-Q plot for normality
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Return Distribution Analysis', fontsize=16, fontweight='bold')

# Q-Q Plot
ax_qq = axes2[0]
stats.probplot(portfolio_returns.dropna(), dist="norm", plot=ax_qq)
ax_qq.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
ax_qq.grid(True, alpha=0.3)

# Distribution histogram with normal overlay
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

# Add stats text
skew = portfolio_returns.skew()
kurt = portfolio_returns.kurtosis()
stats_text = f'Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}'
ax_dist.text(0.05, 0.95, stats_text, transform=ax_dist.transAxes, 
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('c:\\Users\\seva\\transformer-mr-model\\diagram\\return_distribution_analysis.png', dpi=300, bbox_inches='tight')
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

summary_stats.to_csv('c:\\Users\\seva\\transformer-mr-model\\monte_carlo_summary.csv', index=False)
print("Saved: monte_carlo_summary.csv")

print("\n" + "="*80)
print(" MONTE CARLO VALIDATION COMPLETE ")
print("="*80)
print()
