# --------------
# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Code starts here
data = pd.read_csv(path)
print(data.head())
data.shape
data.describe
data = data.drop('Serial Number',1)


# code ends here




# --------------
#Importing header files
from scipy.stats import chi2_contingency
import scipy.stats as stats

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 11)   # Df = number of variable categories(in purpose) - 1

# Code starts here
prob = 0.95
return_rating = data["morningstar_return_rating"].value_counts()
risk_rating = data["morningstar_risk_rating"].value_counts()
observed = pd.concat([return_rating.transpose() ,risk_rating.transpose()] , axis=1 , keys= ['return','risk'])
print(observed)

chi2 , p, dof, ex = chi2_contingency(observed)
if abs(chi2) >= critical_value:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')


# Code ends here


# --------------
# Code starts here
correlation = abs(data.corr())
#print(correlation)
us_correlation = correlation.unstack()
#us_correlation = pd.DataFrame(us_correlation)
us_correlation = us_correlation.sort_values(ascending = False)
max_correlated = us_correlation[(us_correlation>0.75) & (us_correlation<1)]
# print(max_correlated, np.unique(max_correlated.index.get_level_values(0)))
# data = data.drop(np.unique(max_correlated.index.get_level_values(0)), 1)
#print(data[max_correlated.index])
data.drop('morningstar_rating',1, inplace=True)
data.drop('portfolio_stocks',1, inplace=True)
data.drop('category_12',1, inplace=True)
# data.drop('morningstar_return_rating',1)
# data.drop('sharpe_ratio_3y',1)
# data.drop('sharpe_ratio_3y',1)
data.drop('sharpe_ratio_3y',1, inplace=True)
# data.drop('sharpe_ratio_3y',1)
# code ends here


# --------------
# Code starts here
f, (ax_1, ax_2) = plt.subplots(1, 2)
ax_1.boxplot(data['price_earning'])
ax_1.set_title('price_earning')
ax_2.boxplot(data["net_annual_expenses_ratio"])
ax_2.set_title('net_annual_expenses_ratio')

# code ends here


# --------------
# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
# Code starts here
X = data.drop('bonds_aaa',1)
y = data['bonds_aaa'].copy()
X_train,X_test,y_train,y_test =train_test_split(X,y, test_size = 0.3 , random_state = 3)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
rmse = sqrt(mean_squared_error(y_test,y_pred))
print("The RMSE of model is:",round(rmse))
# Code ends here


# --------------
# import libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.linear_model import LinearRegression
# regularization parameters for grid search
ridge_lambdas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
lasso_lambdas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]

# Code starts here
regressor = LinearRegression()
ridge_model = Ridge(random_state = 0)
ridge_grid = GridSearchCV(estimator=ridge_model, param_grid=dict(alpha=ridge_lambdas))
ridge_grid.fit(X_train,y_train)
ridge_pred = ridge_grid.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(ridge_pred,y_test))
print("Ridge RMSE:",ridge_rmse)

lasso_model = Lasso(random_state = 0)
lasso_grid = GridSearchCV(estimator=lasso_model, param_grid=dict(alpha=lasso_lambdas))
lasso_grid.fit(X_train,y_train)
lasso_pred = lasso_grid.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(lasso_pred,y_test))
print("Lasso RMSE:",lasso_rmse)


# Code ends here


