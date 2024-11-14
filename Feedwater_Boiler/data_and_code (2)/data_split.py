#import pandas as pd

# Read XLSX file into DataFrame
#df = pd.read_excel('playground.xlsx')

# Get rows where 'b' column is 100
##b_100_rows = df[df['b'] == 100]

# Group by consecutive sequences of 100 values
#b_100_groups = b_100_rows.groupby(b_100_rows.index - b_100_rows.index.searchsorted(b_100_rows.index))

# Write each group to a separate CSV file
#for i, group in enumerate(b_100_groups, start=1):
#    group[1].to_csv(f"{i}.csv", index=False)
import pandas as pd

# Read 49.csv into DataFrame
df = pd.read_csv('49.csv')

# Create a new column to store the grouping
df['group_status'] = 0

# Iterate through the DataFrame in groups of 10 rows
for i in range(0, len(df)-9, 10):
    group = df.iloc[i:i+10]
    next_group = df.iloc[i+10:i+20]
    
    # Check if there is a 1 in the next 10 rows' 'o' column
    if 1 in next_group['o'].values:
        df.loc[group.index, 'group_status'] = 1

# Write updated DataFrame to new CSV file
df.to_csv('49_updated.csv', index=False)