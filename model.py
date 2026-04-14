import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Population Parameters
POISSON_LAMBDA = 50       # Mean daily demand
MIN_LEAD_TIME = 3         # Minimum delivery days
MAX_LEAD_TIME = 6         # Maximum delivery days
NUM_SAMPLES = 100000      # Size of our simulated Sampling

# 1. Generating Data using specific Discrete Distributions
# Demand: Poisson Distribution
daily_demand = np.random.poisson(lam=POISSON_LAMBDA, size=NUM_SAMPLES)

# Lead Time: Discrete Uniform Distribution
lead_times = np.random.randint(low=MIN_LEAD_TIME, high=MAX_LEAD_TIME + 1, size=NUM_SAMPLES)

# 2. Multiple Discrete Random Variables: Demand During Lead Time
ddlt_array = daily_demand * lead_times

# 3. Sample Statistics
sample_mean = np.mean(ddlt_array)
sample_std = np.std(ddlt_array)

# 4. Standard Normal Distribution for the threshold
# We target a 95% probability of not running out of stock
confidence_level = 0.95
z_score = stats.norm.ppf(confidence_level) # Standard Normal Z-score

# 5. Mathematical Expectation & Final Calculation
safety_stock = z_score * sample_std
reorder_point = sample_mean + safety_stock

# Output Results
print(f"Sample Mean (Expected DDLT): {sample_mean:.0f} units")
print(f"Sample Standard Deviation: {sample_std:.0f} units")
print(f"Standard Normal Z-Score: {z_score:.3f}")
print(f"Final Reorder Point: {reorder_point:.0f} units")

# Visualization: Plotting the Normal Distribution fit
plt.figure(figsize=(8, 5))
count, bins, ignored = plt.hist(ddlt_array, bins=30, density=True, alpha=0.6, color='b', label='Sample Data')

# Overlay Continuous Normal Distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, sample_mean, sample_std)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')

plt.axvline(reorder_point, color='r', linestyle='dashed', linewidth=2, label=f'Reorder Point (Z={z_score:.2f})')
plt.title('Demand During Lead Time: Normal Approximation')
plt.xlabel('Units')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('inventory_distribution.png')