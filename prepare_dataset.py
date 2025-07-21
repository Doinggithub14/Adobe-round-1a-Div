import os
import json
import pandas as pd

data = []

# Loop through all ground_truth JSON files
for filename in os.listdir('ground_truth'):
    if filename.endswith('.json'):
        filepath = os.path.join('ground_truth', filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            for item in content.get("outline", []):
                data.append({
                    "text": item["text"],
                    "level": item["level"]
                })

# Convert list to CSV
df = pd.DataFrame(data)
df.to_csv('headings_dataset.csv', index=False)

print("✅ CSV created: headings_dataset.csv")