import pandas as pd

df = pd.read_csv("results.csv")
df = df.round(decimals=0)

df.to_csv('resultstemp.csv', index=False)