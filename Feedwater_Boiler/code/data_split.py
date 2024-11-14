import pandas as pd

# Read XLSX file into DataFrame
df = pd.read_excel('your_file.xlsx')

# Get rows where 'b' column is 100
b_100_rows = df[df['b'] == 100]

# Group by consecutive sequences of 100 values
b_100_groups = b_100_rows.groupby(b_100_rows.index - b_100_rows.index.searchsorted(b_100_rows.index))

# Write each group to a separate CSV file
for i, group in enumerate(b_100_groups, start=1):
    group[1].to_csv(f"{i}.csv", index=False)