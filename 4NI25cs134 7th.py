#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv(r"C:\Users\user\Downloads\archive (12)\Student_Performance.csv")
print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())


# In[8]:


df["Extracurricular Activities"]=df["Extracirricular Activities"] =df["Extracurricular Activities"].map({"Yes":1,"No":0})


# In[10]:


plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("01_correlation.png")
plt.show()


# In[14]:


X_simple = df[["Hours Studied"]]
y = df ["Performance Index"]

xs_train, Xs_test, ys_train, ys_test = train_test_split(
X_simple, y, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(xs_train , ys_train)
print("\n=== SIMPLE LINEAR REGRESSION ===")
print("Intercept (b0): ", simple_model.intercept_)
print("Slope (b1)    : ", simple_model.coef_[0])

ys_pred = simple_model.predict(Xs_test)
print("R2     : ", r2_score(ys_test, ys_pred))
print("RMSE   : ", np.sqrt(mean_squared_error(ys_test, ys_pred)))


# In[15]:


# Plot the fitted line
plt.figure(figsize=(7, 5))
plt.scatter(Xs_test, ys_test, alpha=0.4, label="Actual")
plt.plot(Xs_test.sort_values("Hours Studied"), 
         simple_model.predict(Xs_test.sort_values("Hours Studied")), 
         color="red", label="Fitted Line")
plt.xlabel("Hours Studied")
plt.ylabel("Performance Index")
plt.legend()
plt.title("Simple Linear Regression")
plt.tight_layout()
plt.savefig("02_simple_fit.png")
plt.show()


# In[18]:


# ------------------------------------------------------------
# 4. MULTIPLE LINEAR REGRESSION (many features -> 1 target)
# ------------------------------------------------------------
X = df.drop(columns="Performance Index")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

print("\n=== MULTIPLE LINEAR REGRESSION ===")
print("Intercept: ", multi_model.intercept_)
for name, coef in zip(X.columns, multi_model.coef_):
    print(f"  {name:35s} {coef: .4f}")

y_pred = multi_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
n, p = X_test.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("R2           :", r2)
print("Adjusted R2  :", adj_r2)
print("RMSE         :", rmse)


# In[20]:


# 5. CHECKING ASSUMPTIONS (use statsmodels on the training data)
# -------------------------------------------------------------
X_train_sm = sm.add_constant(X_train)       # statsmodels needs the intercept column
ols = sm.OLS(y_train, X_train_sm).fit()     # coefficients, p-values, R2, etc.
print("\n", ols.summary())

fitted = ols.fittedvalues
resid = ols.resid

# 5a. Linearity + Homoscedasticity: residuals vs fitted
plt.figure(figsize=(7, 5))
plt.scatter(fitted, resid, alpha=0.3)
plt.axhline(0, color="red")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (want a random cloud around 0)")
plt.tight_layout()
plt.savefig("03_resid_vs_fitted.png")
plt.show()


# In[24]:


# 5b. Normality of residuals: Q-Q plot + histogram
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sm.qqplot(resid, line="45", fit=True, ax=ax[0])
ax[0].set_title("Q-Q plot (want points on the line)")
sns.histplot(resid, kde=True, ax=ax[1])
ax[1].set_title("Residual histogram (want bell shape)")
plt.tight_layout()
plt.savefig("04_normality.png")
plt.show()


# In[22]:


# 5c. Homoscedasticity test: Breusch-Pagan (p > 0.05 => constant variance ok)
bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_train_sm)
print("Breusch-Pagan p-value:", bp_p)


# In[23]:


# 5d. Independence of errors: Durbin-Watson (~2 means no autocorrelation)
print("Durbin-Watson:", durbin_watson(resid))


# In[25]:


# 5e. Multicollinearity: VIF (want < 5, worry above 10)
vif = pd.DataFrame({
    "Feature": X_train_sm.columns,
    "VIF": [variance_inflation_factor(X_train_sm.values, i)
            for i in range(X_train_sm.shape[1])]
})

print("\nVIF:\n", vif[vif["Feature"] != "const"])


# In[26]:


plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")  # perfect-predict
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (closer to red line = better)")
plt.tight_layout()
plt.savefig("05_actual_vs_pred.png")
plt.show()


# In[ ]:




