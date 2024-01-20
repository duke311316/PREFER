# Python program to convert
# JSON file to CSV

import pandas as pd
import json

with open('KLALAFAY22-20230802.json', 'r') as f:
    data = json.load(f)
df = pd.json_normalize(data)
df.to_csv('KLALAFAY22-20230802.csv')
