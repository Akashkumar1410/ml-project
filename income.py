import pandas as pd
from pandas_profiling import ProfileReport

# Assuming 'df_copy1' is your DataFrame
# If you already have a DataFrame, replace this with your actual DataFrame
df_copy1 = pd.read_csv('new_income_file.csv')

# Generate a data profiling report
profile = ProfileReport(df_copy1, title='Data Profiling Report', explorative=True)

# Save the report to an HTML file
profile.to_file("data_profiling_report.html")
