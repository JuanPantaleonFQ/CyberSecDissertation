import pandas as pd

# Load all parts and concatenate them
files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

print("Dataset shape:", df.shape)
df.head()
