import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. System Parameters
NUM_SAMPLES = 10000
MEAN_DEMAND = 50
STD_DEMAND = 12
MEAN_LEAD_TIME = 4
STD_LEAD_TIME = 1.5

# 2. Cost Function Parameters (Newsvendor Model)
SHORTAGE_COST = 40  # Profit lost per stockout
HOLDING_COST = 5    # Cost to store one unit
critical_ratio = SHORTAGE_COST / (SHORTAGE_COST + HOLDING_COST)

# 3. Data Simulation
demand_data = np.maximum(np.random.normal(MEAN_DEMAND, STD_DEMAND, NUM_SAMPLES), 0)
lead_time_data = np.maximum(np.random.normal(MEAN_LEAD_TIME, STD_LEAD_TIME, NUM_SAMPLES), 1)

# Export sample data for the assignment table
df = pd.DataFrame({
    'Daily_Demand_Units': np.round(demand_data).astype(int),
    'Lead_Time_Days': np.round(lead_time_data).astype(int)
})
df.head(20).to_csv('presentation_sample_data.csv', index=False)

# 4. Statistical Aggregation
mu_D, var_D = np.mean(demand_data), np.var(demand_data)
mu_L, var_L = np.mean(lead_time_data), np.var(lead_time_data)

# Demand During Lead Time (DDLT)
expected_ddlt = mu_L * mu_D
var_ddlt = (mu_L * var_D) + ((mu_D ** 2) * var_L)
std_ddlt = np.sqrt(var_ddlt)

# 5. Threshold Calculation
z_score = stats.norm.ppf(critical_ratio)
safety_stock = z_score * std_ddlt
reorder_point = expected_ddlt + safety_stock

# Print output for verification
print(f"Target Service Level (Critical Ratio): {critical_ratio:.1%}")
print(f"Expected DDLT: {expected_ddlt:.0f} units")
print(f"Safety Stock: {safety_stock:.0f} units")
print(f"Optimal Reorder Point: {reorder_point:.0f} units")

# 6. Presentation Visualization
plt.figure(figsize=(10, 6))

# Generate the theoretical normal curve for DDLT
x_axis = np.linspace(expected_ddlt - 4*std_ddlt, expected_ddlt + 4*std_ddlt, 1000)
y_axis = stats.norm.pdf(x_axis, expected_ddlt, std_ddlt)
plt.plot(x_axis, y_axis, color='black', linewidth=2, label='DDLT Distribution')

# Shade the safe zone (Area under curve up to ROP)
x_fill = np.linspace(expected_ddlt - 4*std_ddlt, reorder_point, 1000)
y_fill = stats.norm.pdf(x_fill, expected_ddlt, std_ddlt)
plt.fill_between(x_fill, y_fill, color='lightblue', alpha=0.6, label=f'Safe Zone ({critical_ratio:.1%} Probability)')

# Add threshold markers
plt.axvline(expected_ddlt, color='gray', linestyle='--', linewidth=2, label=f'Expected Demand ({expected_ddlt:.0f})')
plt.axvline(reorder_point, color='red', linestyle='-', linewidth=2.5, label=f'Reorder Point ({reorder_point:.0f})')

# Formatting for presentation readability
plt.title('Optimal Inventory Threshold (Demand During Lead Time)', fontsize=16, fontweight='bold')
plt.xlabel('Units Demanded During Delivery Window', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('inventory_optimization_plot.png', dpi=300)
print("Visualization saved as 'inventory_optimization_plot.png'.")