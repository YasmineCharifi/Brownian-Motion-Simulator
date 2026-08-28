"""
Chebychev Bound vs Empirical Tail Probability
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Chebychev's Inequality is represented as : P(|X-mu|> k*sigma) <= 1/k^2
# This bound holds for any distribution with finite variance, so we will compare this universal
# bound against actual tail probabilities
#
distributions = {
    'Normal': lambda n : np.random.normal(0,1),
    'Uniform': lambda n : np.random.uniform(-np.sqrt(3), np.sqrt(3), n),
    'Exponential': lambda n : np.random.exponential(1,n)-1,
    'Laplace': lambda n : np.random.laplace(0,1/np.sqrt(2),n),
    'Student-t (df=5)': lambda n : np.random.standard_t(5,n)*np.sqrt(3/5),
    'Beta (0.5,0.5)': lambda n : (np.random.beta(0.5,0.5,n)-0.5)*4,
} 
n_samples= 100000
k_values = np.linspace(1,5,50)
chebychev_bound = 1/k_values**2

# compute empirical probabilities
empirical_probs = {}
for name, sampler in distributions.items():
    samples = sampler(n_samples)
    mu, sigma = np.mean(samples), np.std(samples)
    probs = [np.mean(np.abs(samples-mu)>=k*sigma) for k in k_values]
    empirical_probs[name] = probs

# create a subplot visuaization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, probs) in enumerate(empirical_probs.items()):
    ax = axes[idx]
    ax.plot(k_values, chebychev_bound, 'r--', lw=2, label='Chebychev(1/k^2)')
    ax.plot(k_values, probs, 'b-', lw=2, label='Empirical')
    ax.fill_between(k_values, probs, chebychev_bound, alpha=0.3, color='red')
    ax.set_xlabel('k (std deviations)')
    ax.set_ylabel('P(|X-mu|>= k*sigma)')
    ax.set_title(name, fontweight='bold')
    ax.legend()
    ax.set_ylim(0,1)
    ax.grid(True, alpha=0.3)

plt.suptitle('Chebychev Bound vs Empirical Tail Probabilities', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/chebychev_comparison.png',dpi=150, bbox_inches='tight')
plt.show()

# Print comparison table
print('\nTail probability Comparison')
print('='*60)
print(f"{'Distribution':<20} {'k=2':>10} {'k=3':>10}{'k=3':>10}")
print('='*60)
print(f"{'Chebychev Bound':<20} {0.25:>10.4f} {0.111:>10.4f} {0.0625:>10.4f}")
for name, probs in empirical_probs.items():
    p2 = probs[np.argmin(np.abs(k_values-2))]
    p3 = probs[np.argmin(np.abs(k_values-3))]
    p4 = probs[np.argmin(np.abs(k_values-4))]
    print(f"{name:<20} {p2:>10.4f} {p3:>10.4f} {p4:>10.4f}")


