#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is a simple plot for comparing original and z-score values of the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import matplotlib.pyplot as plt
import sys

if sys.stdin.isatty():
    print("No data to plot.")
    sys.exit(0)

df = pd.read_csv(sys.stdin)

plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
df['temperature_2m'].hist(bins=20, color='skyblue')
plt.title('Temperature Distribution')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.subplot(1, 4, 2)
df['relative_humidity_2m'].hist(bins=20, color='salmon')
plt.title('Relative Humidity Distribution')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Frequency')
plt.subplot(1, 4, 3)
df['temperature_2m_zscore'].hist(bins=20, color='skyblue')
plt.title('Temperature Z-Score Distribution')
plt.xlabel('Temperature Z-Score')
plt.ylabel('Frequency')
plt.subplot(1, 4, 4)
df['relative_humidity_2m_zscore'].hist(bins=20, color='salmon')
plt.title('Relative Humidity Z-Score Distribution')
plt.xlabel('Relative Humidity Z-Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
