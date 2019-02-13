---
layout: post
title:  "Introduction to Linear Regression With Python"
date:   2019-02-13 15:20:34 -0500
categories: [Machine Learning]
tags: [python, machine-learning, linear-regression]
---

Using the Auto dataset

Reference: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Auto.html


```python
# Common imports
import numpy as np
import os
```


```python
# Keep a same seed in different executions
np.random.seed(42)
```


```python
# More common imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
```


```python
# Directory path of datasets
datapath = os.path.join("datasets", "islr", "")
```


```python
# Load dataset into a dataframe
auto = pd.read_csv(datapath + "Auto.data.txt", delim_whitespace=True, na_values='?')
auto.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Query NaN values
auto[auto.isnull().any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>25.0</td>
      <td>4</td>
      <td>98.0</td>
      <td>NaN</td>
      <td>2046.0</td>
      <td>19.0</td>
      <td>71</td>
      <td>1</td>
      <td>ford pinto</td>
    </tr>
    <tr>
      <th>126</th>
      <td>21.0</td>
      <td>6</td>
      <td>200.0</td>
      <td>NaN</td>
      <td>2875.0</td>
      <td>17.0</td>
      <td>74</td>
      <td>1</td>
      <td>ford maverick</td>
    </tr>
    <tr>
      <th>330</th>
      <td>40.9</td>
      <td>4</td>
      <td>85.0</td>
      <td>NaN</td>
      <td>1835.0</td>
      <td>17.3</td>
      <td>80</td>
      <td>2</td>
      <td>renault lecar deluxe</td>
    </tr>
    <tr>
      <th>336</th>
      <td>23.6</td>
      <td>4</td>
      <td>140.0</td>
      <td>NaN</td>
      <td>2905.0</td>
      <td>14.3</td>
      <td>80</td>
      <td>1</td>
      <td>ford mustang cobra</td>
    </tr>
    <tr>
      <th>354</th>
      <td>34.5</td>
      <td>4</td>
      <td>100.0</td>
      <td>NaN</td>
      <td>2320.0</td>
      <td>15.8</td>
      <td>81</td>
      <td>2</td>
      <td>renault 18i</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Clean data
auto = auto.dropna()
# Query data types of all columns
auto.dtypes
```




    mpg             float64
    cylinders         int64
    displacement    float64
    horsepower      float64
    weight          float64
    acceleration    float64
    year              int64
    origin            int64
    name             object
    dtype: object




```python
# Select mpg as output(y) and horsepower as one feature(X)
# X and Y become matrix shape
X = np.c_[auto['horsepower']]
y = np.c_[auto['mpg']]
print(X.shape)
print(y.shape)
```

    (392, 1)
    (392, 1)
    


```python
# Plot x vs y by selection two columns of the dataframe
auto.plot(kind='scatter', x="horsepower", y='mpg')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2ae2bfb780>




![png](/assets/2019-02-13-linear-regression-intro/output_10_1.png)



```python
# Define a linear regression model
model = sklearn.linear_model.LinearRegression()
```


```python
# Fit our linear regression model
model.fit(X,y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
# Make a prediction for 150 horsepower
X_sample = np.array([150]).reshape(1,1) # 
print(model.predict(X_sample)) # 
```

    [[16.25915102]]
    


```python
# turn the car model name into index
auto.set_index("name", inplace = True)
auto.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>chevrolet chevelle malibu</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>buick skylark 320</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>plymouth satellite</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>amc rebel sst</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ford torino</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select our data of interest
data = auto[['horsepower','mpg']]
data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>horsepower</th>
      <th>mpg</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>chevrolet chevelle malibu</th>
      <td>130.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>buick skylark 320</th>
      <td>165.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>plymouth satellite</th>
      <td>150.0</td>
      <td>18.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Show three cases on our plot
data.plot(kind='scatter', x="horsepower", y='mpg', figsize=(10,6))
plt.axis([0, 250, 0, 50])
text_pos = { "vw pickup": (20, 35), "ford ranger": (95, 36), "bmw 2002": (130, 30)}
for carmodel, text_pos_i in text_pos.items():
    pos_x, pos_y = data.loc[carmodel]
    plt.annotate(carmodel, xy=(pos_x, pos_y), xytext=text_pos_i,
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_x, pos_y, "ro")
plt.show()
```


![png](/assets/2019-02-13-linear-regression-intro/output_16_0.png)



```python
# Plot two hypothesis
import numpy as np

data.plot(kind='scatter', x="horsepower", y='mpg', figsize=(10,6))
plt.axis([0, 250, 0, 50])
X=np.linspace(0, 250, 1000)
plt.plot(X, 45 - X/5, "r")
plt.text(10, 35, r"$\theta_0 = 45$", fontsize=14, color="r")
plt.text(10, 31, r"$\theta_1 = 0.2 \times 10^{-1}$", fontsize=14, color="r")

plt.plot(X, 20 + 0 * X, "b")
plt.text(200, 25, r"$\theta_0 = 20$", fontsize=14, color="b")
plt.text(200, 22, r"$\theta_1 = 0$", fontsize=14, color="b")
plt.show()
```


![png](/assets/2019-02-13-linear-regression-intro/output_17_0.png)



```python
# Obtain one good linear hypothesis
from sklearn import linear_model
linearModel = linear_model.LinearRegression()
X = np.c_[data["horsepower"]]
y = np.c_[data["mpg"]]
linearModel.fit(X, y)
theta0, theta1 = linearModel.intercept_[0], linearModel.coef_[0][0]
theta0, theta1
```




    (39.93586102117047, -0.15784473335365365)




```python
# Plot our estimated good linear hypothesis
data.plot(kind='scatter', x="horsepower", y='mpg', figsize=(10,6))
plt.axis([0, 250, 0, 50])
X=np.linspace(0, 250, 1000)
plt.plot(X, theta0 + theta1*X, "b")
plt.text(5, 30, r"$\theta_0 = 39.93 $", fontsize=14, color="b")
plt.text(5, 27, r"$\theta_1 = 15.78 \times 10^{-2}$", fontsize=14, color="b")
plt.show()
```


![png](/assets/2019-02-13-linear-regression-intro/output_19_0.png)



```python
x_test = np.array([150]).reshape(1,1)
y_predited = linearModel.predict(x_test)
y_predited
```




    array([[16.25915102]])




```python
# Draw prediction on our plot
data.plot(kind='scatter', x="horsepower", y='mpg', figsize=(10,6), s=1)
X=np.linspace(0, 250, 1000)
plt.plot(X, theta0 + theta1*X, "b")
plt.axis([0, 250, 0, 50])
plt.text(10, 30, r"$\theta_0 = 39.93 $", fontsize=14, color="b")
plt.text(10, 27, r"$\theta_1 = 15.78 \times 10^{-2}$", fontsize=14, color="b")
plt.plot([x_test[0,0], x_test[0,0]], [0, y_predited], "r--")
plt.text(150, 20, r"Prediction = 16.26", fontsize=14, color="b")
plt.plot(x_test, y_predited, "ro")
plt.show()
```


![png](/assets/2019-02-13-linear-regression-intro/output_21_0.png)



```python
# Separate "nonrepresentative" data points
data_excluded = data[data.horsepower>200]
data_excluded
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>horsepower</th>
      <th>mpg</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>chevrolet impala</th>
      <td>220.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>plymouth fury iii</th>
      <td>215.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>pontiac catalina</th>
      <td>225.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>buick estate wagon (sw)</th>
      <td>225.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>ford f250</th>
      <td>215.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>dodge d200</th>
      <td>210.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>mercury marquis</th>
      <td>208.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>chrysler new yorker brougham</th>
      <td>215.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>buick electra 225 custom</th>
      <td>225.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>pontiac grand prix</th>
      <td>230.0</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_excluded[['horsepower', 'mpg']].values
```




    array([[220.,  14.],
           [215.,  14.],
           [225.,  14.],
           [225.,  14.],
           [215.,  10.],
           [210.,  11.],
           [208.,  11.],
           [215.,  13.],
           [225.,  12.],
           [230.,  16.]])




```python
# Define data_new without "nonrepresentative" data points
data_new = data[data.horsepower<=200]
data_new.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>horsepower</th>
      <th>mpg</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>chevrolet chevelle malibu</th>
      <td>130.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>buick skylark 320</th>
      <td>165.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>plymouth satellite</th>
      <td>150.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>amc rebel sst</th>
      <td>150.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>ford torino</th>
      <td>140.0</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compare a full model to a model without "norepresentative" data points
data_new.plot(kind='scatter', x="horsepower", y='mpg', figsize=(10,6))
plt.axis([0, 250, 0, 50])
text_pos = { "pontiac catalina": (160, 34), "buick estate wagon (sw)": (140, 37), 
            "buick electra 225 custom": (160, 30), "pontiac grand prix": (180,40)}

for model, text_pos_i in text_pos.items():
    pos_x, pos_y = data_excluded.loc[model]
    plt.annotate(model, xy=(pos_x, pos_y), xytext=text_pos_i,
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
#    plt.plot(pos_x, pos_y, "rs")


for xi,yi in data_excluded[['horsepower', 'mpg']].values:
    plt.plot(xi, yi, "rs")

X=np.linspace(0, 250, 1000)
X_new = np.c_[data_new["horsepower"]]
y_new = np.c_[data_new["mpg"]]
linearModel.fit(X_new, y_new)
theta0, theta1 = linearModel.intercept_[0], linearModel.coef_[0][0]
plt.plot(X, theta0 + theta1*X, "b:", label="Linear model on partial data")

lin_reg_full = linear_model.LinearRegression()
Xfull = np.c_[data["horsepower"]]
yfull = np.c_[data["mpg"]]
lin_reg_full.fit(Xfull, yfull)

theta0full, theta1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]
X = np.linspace(0, 250, 1000)
plt.plot(X, theta0full + theta1full * X, "b", label="Linear model on full data")

plt.legend(loc="lower left")

plt.show()
```


![png](/assets/2019-02-13-linear-regression-intro/output_25_0.png)



```python
# Fit a polynomial model
data.plot(kind='scatter', x="horsepower", y='mpg', figsize=(10,6))
plt.axis([0, 250, 0, 50])

from sklearn import preprocessing
from sklearn import pipeline

poly = preprocessing.PolynomialFeatures(degree=4, include_bias=False)
XFullPol = poly.fit_transform(Xfull)

lin_reg2 = linear_model.LinearRegression()
lin_reg2.fit(XFullPol, yfull)

XPol = poly.fit_transform(X[:,np.newaxis])
curve = lin_reg2.predict(XPol)
plt.plot(X, curve)
plt.show()
```


![png](/assets/2019-02-13-linear-regression-intro/output_26_0.png)



```python
# Fit a polynomial model (Another way)
data.plot(kind='scatter', x="horsepower", y='mpg', figsize=(10,6))
plt.axis([0, 250, 0, 50])

from sklearn import preprocessing
from sklearn import pipeline

poly = preprocessing.PolynomialFeatures(degree=4, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression()

pipeline_reg = pipeline.Pipeline([('poly', poly), ('scal', scaler), ('lin', lin_reg2)])
pipeline_reg.fit(Xfull, yfull)
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve)
plt.show()
```


![png](/assets/2019-02-13-linear-regression-intro/output_27_0.png)



```python
# Add a regularized model and compare this one to the previous models
plt.figure(figsize=(10,6))

plt.xlabel("horsepower")
plt.ylabel('mpg')

plt.plot(list(data_new["horsepower"]), list(data_new["mpg"]), "bo")
plt.plot(list(data_excluded["horsepower"]), list(data_excluded["mpg"]), "rs")

X = np.linspace(0, 250, 1000)
plt.plot(X, theta0full + theta1full * X, "r--", label="Linear model on all data")
plt.plot(X, theta0 + theta1*X, "b:", label="Linear model on partial data")


ridge = linear_model.Ridge(alpha=10**5)
Xsample = np.c_[data_new["horsepower"]]
ysample = np.c_[data_new["mpg"]]
ridge.fit(Xsample, ysample)
theta0ridge, theta1ridge = ridge.intercept_[0], ridge.coef_[0][0]
plt.plot(X, theta0ridge + theta1ridge * X, "b", label="Regularized linear model on partial data")


plt.legend(loc="lower right")
plt.axis([0, 250, 0, 50])
plt.show()
```


![png](/assets/2019-02-13-linear-regression-intro/output_28_0.png)


# References
Hands-On Machine Learning with Scikit-Learn and TensorFlow
http://shop.oreilly.com/product/0636920052289.do
