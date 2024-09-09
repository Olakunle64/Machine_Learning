from sklearn.datasets import fetch_openml
import pandas as pd

# Fetch the complete dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')

# Convert to DataFrame
mnist_df = pd.DataFrame(data=mnist['data'], columns=mnist['feature_names'])
mnist_df['target'] = mnist['target']

# Take a random sample of 10,000 rows
mnist_sample = mnist_df.sample(n=10000, random_state=42)
mnist_sample.to_csv('mnist_sample.csv', index=False)

print("Sample of 1,000 rows saved to 'mnist_sample.csv'.")