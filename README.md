# aid_allocation
Data analytics project investigating the allocation of UK foreign aid

# pre-process.py
Data preparation: Read data from csv and excel files.  Filter data. Merge into one DataFrame.
Handle missing values.  
Data transformation: Perform log transformations.  Handle outliers.  Standardise using MinMaxScaler()

# figures.py
Data transformation: plot histograms to show the original vs. the log transformation 
Data exploration: 
  plot histogram to show distribution of the target variable aid_percentage 
  define aid_bool and plot pie chart to show % of countries who are recipients of aid
  plot scatter plots to show the relationship between each feature and the target variable

# regression.py
Feature selection:
  plot correlation matrix
  perform feature selectino using SelectKBest with score_func = f_regression
  perform and evaluate linear regression
