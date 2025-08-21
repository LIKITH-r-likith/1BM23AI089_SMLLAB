#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data = {
    "shear": [2160.70, 1680.15, 2318.00, 2063.30, 2209.30, 2209.50, 1710.30, 1786.70,
              2577.90, 2359.90, 2258.70, 2167.20, 2401.55, 1781.80, 2338.75, 1767.30,
              2055.50, 2416.40, 2202.50, 2656.20, 1755.70],
    "age": [15.50, 23.75, 8.00, 17.00, 5.50, 19.00, 24.00, 2.50, 7.50, 11.00, 13.00,
            3.75, 25.00, 9.75, 22.00, 18.00, 6.00, 12.50, 2.00, 21.50, 0.00]
}

df = pd.DataFrame(data)
y = df['shear']
X = df['age']
X = sm.add_constant(X) 

linear_regression = sm.OLS(y, X)
fitted_model = linear_regression.fit()
print(fitted_model.summary())
intercept = fitted_model.params['const']
slope = fitted_model.params['age']
print("\nIntercept:", intercept)
print("Slope:", slope)
plt.scatter(df['age'], df['shear'], label='Data Points')
plt.plot(df['age'], intercept + slope * df['age'], color='red', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Shear Strength')
plt.title('Linear Regression: Shear vs Age')
plt.legend()
plt.grid(True)
plt.show()


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "shear": [2160.70, 1680.15, 2318.00, 2063.30, 2209.30, 2209.50, 1710.30, 1786.70,
              2577.90, 2359.90, 2258.70, 2167.20, 2401.55, 1781.80, 2338.75, 1767.30,
              2055.50, 2416.40, 2202.50, 2656.20, 1755.70],
    "age": [15.50, 23.75, 8.00, 17.00, 5.50, 19.00, 24.00, 2.50, 7.50, 11.00, 13.00,
            3.75, 25.00, 9.75, 22.00, 18.00, 6.00, 12.50, 2.00, 21.50, 0.00]
}

df = pd.DataFrame(data)
x = df['age'].values
y = df['shear'].values

x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)
slope = numerator / denominator
intercept = y_mean - slope * x_mean
print("Intercept:", intercept)
print("Slope:", slope)
y_pred = intercept + slope * x

plt.scatter(x, y, label='Data Points')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Shear Strength')
plt.title('Manual Linear Regression: Shear vs Age')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




