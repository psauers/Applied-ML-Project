import pandas as pd
import sys

# Get data path parameter
data_path = sys.argv[1]

# Load data 
df = pd.read_csv(data_path)

# Split original data set into 2 smaller sets: numeric and categorical features
df_categ = df.select_dtypes(include=['object'])
df_numeric = df.select_dtypes(include=['int64'])

# Get dataset information
print(df.info())

# Descriptive statistics of numeric features
print(df_numeric.describe())

# Create correlation matrix
print(df.corr())

# Create boxplots for numeric categorical
for i in list(df_numeric.columns):
    df.boxplot(column = i, by = 'deposit', grid = False)

# Descriptive statistics of categorical features
print(df_categ.describe())

# Generate stacked bar charts for categorical features
for i in list(df_categ.columns):
    df.groupby([i, 'deposit'])['deposit'].count().unstack().plot.bar(stacked=True)