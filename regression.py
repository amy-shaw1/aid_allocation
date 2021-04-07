import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# *********** feature selection ***********
# define x and y
x = pd.read_csv('scaled_features_df.csv', index_col=0)
y = pd.read_csv('targets_df.csv', index_col=0)

# correlation matrix
corr = x.corr()
fig5 = plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap = "BuPu")
plt.xticks(rotation=90)
plt.savefig('fig5_correlation.png')

# feature selection
kbest = SelectKBest(score_func = f_regression, k = 'all')
ordered_features = kbest.fit(x,y)
df_scores = pd.DataFrame(ordered_features.scores_, columns=['score'])
df_col_names = pd.DataFrame(x.columns, columns=['feature'])
feature_rank = pd.concat([df_col_names, df_scores], axis=1)
feature_rank.to_csv('kbest.csv')

# drop highly correlated columns
x = x.drop(["civil_lib"], axis=1)
# drop poorly performing features
x = x.drop(["mil_exp"], axis=1)

# *********** linear regression ***********
def evaluate(model):
    print("Coefficients: ",model.coef_)
    # cross_val to take 5 different splits of the data, fit a model and compute the score
    MSE_score = cross_val_score(model,x,y,cv=5,scoring="neg_mean_squared_error")
    r2_score = cross_val_score(model,x,y,cv=5,scoring="r2")
    # The best value is 0.0
    print('mean_squared_error: ', MSE_score)
    print('mean_squared_error mean (s.d.): ', np.mean(MSE_score),"(", np.std(MSE_score),")")
    # The best possible score is 1.0 and it can be negative
    print('r2: ', r2_score)
    print('r2 Mean (s.d.): ', np.mean(r2_score),"(", np.std(r2_score),")")

# split 20% as test and 80% as training data
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state=0)

# Create linear regression object
linreg = linear_model.LinearRegression()
# fit to model
linreg.fit(x_train,y_train)
evaluate(linreg)

# model and plot each feature
fig6a, ax6a = plt.subplots(2,2)
count = 0
for feature_name in x.columns[:4]:
    xi_train = x_train[feature_name][:,np.newaxis]
    xi_test = x_test[feature_name][:,np.newaxis]
    linreg.fit(xi_train,y_train)
    y_predicted = linreg.predict(xi_test)
    ax = ax6a.flatten()[count]
    ax.scatter(xi_test, y_test)
    ax.plot(xi_test,y_predicted)
    ax.set_title(feature_name)
    count += 1
plt.tight_layout()
plt.savefig('fig6a_feature_regression.png')

fig6b, ax6b = plt.subplots(2,2)
count = 0
for feature_name in x.columns[4:]:
    xi_train = x_train[feature_name][:,np.newaxis]
    xi_test = x_test[feature_name][:,np.newaxis]
    linreg.fit(xi_train,y_train)
    y_predicted = linreg.predict(xi_test)
    ax = ax6b.flatten()[count]
    ax.scatter(xi_test, y_test)
    ax.plot(xi_test,y_predicted)
    ax.set_title(feature_name)
    count += 1
plt.tight_layout()
plt.savefig('fig6b_feature_regression.png')




