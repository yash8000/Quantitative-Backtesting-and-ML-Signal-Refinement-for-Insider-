import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
df = pd.read_csv('nse_non_derivatives_clean.csv')

# Filter for promoter/promoter group buys
promoter_buys = df[
    (df['person_category'].isin(['Promoters', 'Promoter Group'])) &
    (df['transaction_type'] == 'Buy')
]

# Convert value_traded to numeric, handling any non-numeric values
promoter_buys['value_traded'] = pd.to_numeric(promoter_buys['value_traded'], errors='coerce')

# Basic statistics
stats = promoter_buys['value_traded'].describe()
print("\nValue Traded Statistics (in Rupees):")
print("====================================")
for stat, value in stats.items():
    print(f"{stat:15} : ₹{value:,.2f}")

# Create distribution plot
plt.figure(figsize=(15, 8))

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Regular distribution plot
sns.histplot(data=promoter_buys, x='value_traded', bins=50, ax=ax1)
ax1.set_title('Distribution of Value Traded')
ax1.set_xlabel('Value Traded (₹)')
ax1.set_ylabel('Count')

# Log scale distribution
sns.histplot(data=promoter_buys[promoter_buys['value_traded'] > 0], 
            x='value_traded', bins=50, ax=ax2)
ax2.set_xscale('log')
ax2.set_title('Distribution of Value Traded (Log Scale)')
ax2.set_xlabel('Value Traded (₹) - Log Scale')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.savefig('plots/value_traded_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Print value ranges and their frequencies
print("\nValue Traded Ranges:")
print("===================")
ranges = [
    (0, 100000, "0-1L"),
    (100000, 1000000, "1L-10L"),
    (1000000, 10000000, "10L-1Cr"),
    (10000000, 100000000, "1Cr-10Cr"),
    (100000000, float('inf'), ">10Cr")
]

for start, end, label in ranges:
    count = len(promoter_buys[(promoter_buys['value_traded'] >= start) & (promoter_buys['value_traded'] < end)])
    percentage = (count / len(promoter_buys)) * 100
    print(f"{label:10} : {count:5} trades ({percentage:5.1f}%)")

# Print top 10 largest trades
print("\nTop 10 Largest Trades:")
print("====================")
top_10 = promoter_buys.nlargest(10, 'value_traded')[['symbol', 'value_traded', 'broadcast_datetime']]
for _, row in top_10.iterrows():
    print(f"₹{row['value_traded']:,.2f} - {row['symbol']} ({row['broadcast_datetime']})") 