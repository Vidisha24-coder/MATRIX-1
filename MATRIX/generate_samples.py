"""Utility script to generate sample datasets for Matrix Reconstruction."""
import os
from core.data_loader import (
    generate_sensor_network_matrix,
    generate_collaborative_ratings_matrix,
    generate_financial_metrics_matrix
)

os.makedirs("data", exist_ok=True)

df1, _ = generate_sensor_network_matrix(n_timesteps=30, n_sensors=10, seed=42)
df1.to_csv("data/sample_sensor_network.csv", index=False)

df2, _ = generate_collaborative_ratings_matrix(n_users=20, n_items=8, seed=101)
df2.to_csv("data/sample_customer_ratings.csv", index=False)

df3, _ = generate_financial_metrics_matrix(n_assets=16, n_indicators=7, seed=202)
df3.to_csv("data/sample_financial_metrics.csv", index=False)

print("Generated sample datasets:")
print(" - data/sample_sensor_network.csv")
print(" - data/sample_customer_ratings.csv")
print(" - data/sample_financial_metrics.csv")
