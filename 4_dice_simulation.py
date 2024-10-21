import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Simulate dice rolls
n_rolls = 10000
dice_rolls = np.random.randint(1, 7, size=n_rolls)

# Plot histogram
plt.hist(dice_rolls, bins=6, range=(0.5, 6.5), density=True, alpha=0.7)
plt.title('Dice Roll Distribution')
plt.xlabel('Dice Value')
plt.ylabel('Probability')

# Fit KDE to the data
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(dice_rolls[:, None])

# Plot KDE
x_plot = np.linspace(0.5, 6.5, 1000)[:, None]
log_dens = kde.score_samples(x_plot)
plt.plot(x_plot, np.exp(log_dens), '-', label='KDE')
plt.legend()
plt.show()

# Calculate probabilities
probabilities = np.bincount(dice_rolls)[1:] / n_rolls
for i, prob in enumerate(probabilities, 1):
    print(f"Probability of rolling a {i}: {prob:.4f}")