import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Binomial distribution
n, p = 10, 0.5
x = np.arange(0, n+1)
binomial = stats.binom.pmf(x, n, p)

plt.subplot(2, 1, 1)
plt.bar(x, binomial)
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')

# Normal distribution
mu, sigma = 0, 1
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
normal = stats.norm.pdf(x, mu, sigma)

plt.subplot(2, 1, 2)
plt.plot(x, normal)
plt.title('Normal Distribution (μ=0, σ=1)')
plt.xlabel('Value')
plt.ylabel('Probability Density')

plt.tight_layout()
plt.show()

# Calculate probabilities
print(f"Probability of exactly 5 successes in 10 trials: {stats.binom.pmf(5, n, p):.4f}")
print(f"Probability of value between -1 and 1 in standard normal: {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")