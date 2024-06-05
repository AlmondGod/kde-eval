import pandas as pd

# Load data from the UCI repository
#Aeberhard,Stefan and Forina,M.. (1991). Wine. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(url, header=None)

# Save to CSV
df.to_csv('data/wine.csv', index=False)

# Load data from the UCI repository
# Alpaydin,E. and Kaynak,C.. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'
df = pd.read_csv(url, header=None)

# Save to CSV
df.to_csv('data/optdigits.csv', index=False)

from sklearn.datasets import make_blobs
import pandas as pd

# Generate synthetic data with higher dimensions
X, y = make_blobs(n_samples=1000, centers=5, n_features=50, random_state=42)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
df['Cluster'] = y

print("here")
# Save to CSV
df.to_csv('data/high_dimensional_blobs.csv', index=False)
