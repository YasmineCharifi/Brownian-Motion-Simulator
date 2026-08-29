"""
Monte-Carlo confidence intervals for sample means

The confidence intervals quantify uncertainty in our estimates. Traditional
methods (t-intervals) assume normality, Monte Carlo bootsrap methods are:
- Distribution-free: work regardless of underlying distribution
- Flexible: handle complex statistics beyond just means
- Intuitive: based on simulation
- Robust: don't require analytical formulas

STEPS:
1. Start with your sample size of n
2. resample with replacement: drawing n observations
3. Compute the statistics (mean)
4. Repeat N times
5. Use percentiles of the bootstrap distribution as Confidence Interval bounds
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

def bootstrap_confidence_interval(data, n_bootsrap=10000, confidence_level=0.95):
    """
    Computing bootsrap confidence interval for the mean
    """
    data = np.asarray(data)
    n = len(data)
    original_mean = np.mean(data)

    # generate bootsrap distibution
    bootstrap_means = np.array([
        np.mean(np.random.choice(data, size=n, replace=True))
        for _ in range(n_bootsrap)
    ])
    alpha = 1- confidence_level
    ci_lower = np.percentile(bootstrap_means, 100*alpha/2)
    ci_upper = np.percentile(bootstrap_means, 100*(1-alpha/2))

    return {
        'point_estimate': original_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_distribution': bootstrap_means,
        'standard_error': np.std(bootstrap_means)
    }

#example : income-like skewed data (log-normal)
population = np .random.lognormal(mean=4, sigma=0.8, size=100000)
sample = np.random.choice(population,size=50, replace=False)

print("="*60)
print("Monte Carlo Bootstrap confidence Intervals")
print("="*60)
print(f"\nSample size: {len(sample)}")
print(f"\nSample mean: ${sample.mean():,.2f}")
print(f"\nTrue Population mean: ${population.mean():,.2f}")

# Compute bootstrap CI
result = bootstrap_confidence_interval(sample, n_bootsrap=10000, confidence_level=0.95)

print("\n95% Bootstrap Confidence Interval:")
print(f" Lower: ${result['ci_lower']:,.2f}")
print(f" Upper: ${result['ci_upper']:,.2f}")
print(f" Bootstrap SE: ${result['standard_error']:,.2f}")

# Compare with t-interval
se = stats.sem(sample)
t_crit = stats.t.ppf(0.975, df=len(sample)-1)
t_lower = sample.mean() - t_crit*se
t_upper = sample.mean() + t_crit*se

print("\nClassical t-interval for comparison")
print(f" Lower:${t_lower:,.2f}")
print(f" Upper:${t_upper:,.2f}")


# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14,5))

# original sample distribution
ax1 = axes[0]
ax1.hist(sample, bins=20, edgecolor='black', alpha=0.7,  color='steelblue')
ax1.axvline(sample.mean(), color='red', lw=2, ls='--', label='Sample Mean')
ax1.axvline(population.mean(), color='green', lw=2, ls='--', label='True Mean')
ax1.set_xlabel('value ($)')
ax1.set_ylabel('Frequency')
ax1.set_title('Original Sample (Skewed)', fontweight='bold')
ax1.legend()

# Bootstrap distribution
ax2 = axes[1]
ax2.hist(result['bootstrap_distribution'], bins=20, edgecolor='black', alpha=0.7,  color='coral')
ax2.axvline(result['point_estimate'], color='red', lw=2, ls='--', label='Point Estimate')
ax2.axvline(result['ci_lower'], color='darkred', lw=2, ls=':')
ax2.axvline(result['ci_upper'], color='darkred', lw=2, ls=':', label='95% CI')
ax2.axvline(population.mean(), color='green', lw=2, ls=':', label='True Mean')
ax2.axvspan(result['ci_lower'], result['ci_upper'], alpha=0.2, color='red')
ax2.set_xlabel('Bootstrap sample mean ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Bootstrap Distribution of means', fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig('reports/figures/bootstrap_ci_visualization.png', dpi=150, bbox_inches='tight')
plt.show()



