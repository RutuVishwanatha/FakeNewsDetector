import pandas as pd

# Load fake and real news
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake['label'] = 'FAKE'
true['label'] = 'REAL'

# Combine
df = pd.concat([fake, true], axis=0)
df = df[['text', 'label']]  # Keep only necessary columns

# Save as news.csv
df.to_csv("news.csv", index=False)

print("news.csv created successfully!")
