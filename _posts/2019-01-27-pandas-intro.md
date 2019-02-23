---
layout: post
title:  "Introduction to Pandas"
date:   2019-01-27 09:20:34 -0500
categories: [Python]
tags: [python, pandas, introduction]
---
A brief introduction to pandas


```python
# Import packages
import numpy as np
import pandas as pd
```

### Series
"Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.)."


```python
# Create a Series from a ndarray
s = pd.Series(np.arange(1,20,5))
# Indexes are created by default
s
```




    0     1
    1     6
    2    11
    3    16
    dtype: int32




```python
# Create a Series with indexes
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
```


```python
# Query the indexes of a Series
s.index
```




    Index(['a', 'b', 'c', 'd', 'e'], dtype='object')




```python
# Create a Series from a dict
d = {'k1': 'hello', 'k2': 10, 'k3': 19.8}
pd.Series(d)
```




    k1    hello
    k2       10
    k3     19.8
    dtype: object




```python
# Create a Series from a dict
d = {'k1': 1.0, 'k2': 111, 'k3': 19.8}
pd.Series(d)
```




    k1      1.0
    k2    111.0
    k3     19.8
    dtype: float64




```python
# Create a Series from a scalar value
s = pd.Series(10., index=['a', 'b', 'c'])
s
```




    a    10.0
    b    10.0
    c    10.0
    dtype: float64




```python
# Query the data type
s.dtype
```




    dtype('float64')




```python
# Series to ndarray-like
s.values
```




    array([10., 10., 10.])




```python
# Access values by index values
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s['e'] 
```




    0.22115728602833706




```python
# Access and Set values by index values
s['a'] = 0
s
```




    a    0.000000
    b   -0.378787
    c   -0.380565
    d   -1.015425
    e    0.221157
    dtype: float64




```python
# Query if index exists
'c' in s
```




    True




```python
# Sort values
s.sort_values(ascending=False)
```




    e    0.221157
    a    0.000000
    b   -0.378787
    c   -0.380565
    d   -1.015425
    dtype: float64



### DataFrame
"DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or SQL table, or a dict of Series objects"

"Along with the data, you can optionally pass index (row labels) and columns (column labels) arguments"


```python
# Create a DataFrame from a dict of Series
d = {'first': pd.Series([10, 20, 30], index=['a', 'b', 'c']),
     'second': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
df
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
      <th>first</th>
      <th>second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>10.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>b</th>
      <td>20.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>c</th>
      <td>30.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Query the indexes
df.index
```




    Index(['a', 'b', 'c', 'd'], dtype='object')




```python
# Query the columns
df.columns
```




    Index(['first', 'second'], dtype='object')




```python
# Query the datatypes
df.dtypes
```




    first     float64
    second      int64
    dtype: object




```python
# Create a dataframe picking some indexes
pd.DataFrame(d, index=['c', 'b', 'd'])
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
      <th>first</th>
      <th>second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>30.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>b</th>
      <td>20.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a dataframe picking some columns too
pd.DataFrame(d, index=['c', 'b', 'd'], columns=['second', 'third'])
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
      <th>second</th>
      <th>third</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a dataframe from a dict of ndarrays/list
d = {'first': [10, 20, 30, 40], 'second': [1.1, 1.2, 1.3, 1.4]}
df = pd.DataFrame(d)
df
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
      <th>first</th>
      <th>second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>1.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create dataframe defining indexes
pd.DataFrame(d, index = ['a', 'b', 'c', 'd'])
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
      <th>first</th>
      <th>second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>10</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>b</th>
      <td>20</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>c</th>
      <td>30</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>d</th>
      <td>40</td>
      <td>1.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a dataframe from a NumPy array
df = pd.DataFrame(np.random.randn(10, 4), index=list('abcdefghij'),
             columns=list('ABCD'))
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.396688</td>
      <td>0.311491</td>
      <td>-0.333861</td>
      <td>0.168932</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.525654</td>
      <td>0.834869</td>
      <td>1.636704</td>
      <td>0.393203</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1.461059</td>
      <td>1.011699</td>
      <td>0.089861</td>
      <td>-1.813207</td>
    </tr>
    <tr>
      <th>d</th>
      <td>2.069092</td>
      <td>1.195152</td>
      <td>-1.061165</td>
      <td>-0.777508</td>
    </tr>
    <tr>
      <th>e</th>
      <td>1.453618</td>
      <td>-0.481199</td>
      <td>0.995436</td>
      <td>0.367830</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-1.236010</td>
      <td>-1.037308</td>
      <td>0.981124</td>
      <td>-0.903936</td>
    </tr>
    <tr>
      <th>g</th>
      <td>-0.839587</td>
      <td>0.049658</td>
      <td>-0.431221</td>
      <td>0.603668</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-1.420413</td>
      <td>0.412668</td>
      <td>-0.963789</td>
      <td>0.471481</td>
    </tr>
    <tr>
      <th>i</th>
      <td>0.385292</td>
      <td>0.831554</td>
      <td>-0.781036</td>
      <td>-1.117775</td>
    </tr>
    <tr>
      <th>j</th>
      <td>0.747233</td>
      <td>0.215479</td>
      <td>-0.535672</td>
      <td>-0.231791</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View first rows
df.head()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.260968</td>
      <td>1.462966</td>
      <td>0.935928</td>
      <td>-1.416606</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.158592</td>
      <td>1.765306</td>
      <td>0.153076</td>
      <td>-0.099588</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-3.704010</td>
      <td>0.346566</td>
      <td>1.709313</td>
      <td>3.173983</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-0.454166</td>
      <td>1.719496</td>
      <td>0.095962</td>
      <td>-1.357128</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-2.583424</td>
      <td>-0.319739</td>
      <td>-0.212480</td>
      <td>1.210537</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View last rows
df.tail(3)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>h</th>
      <td>0.526778</td>
      <td>-0.383277</td>
      <td>0.599304</td>
      <td>0.267912</td>
    </tr>
    <tr>
      <th>i</th>
      <td>0.518058</td>
      <td>2.062913</td>
      <td>1.682550</td>
      <td>-0.083072</td>
    </tr>
    <tr>
      <th>j</th>
      <td>-0.603516</td>
      <td>0.332087</td>
      <td>-1.479013</td>
      <td>1.254228</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Return a numpy representation
A = df.values
A
```




    array([[-0.39668765,  0.31149099, -0.33386089,  0.16893183],
           [ 0.5256545 ,  0.83486865,  1.63670404,  0.39320261],
           [ 1.46105881,  1.01169944,  0.08986103, -1.81320698],
           [ 2.06909247,  1.19515187, -1.06116542, -0.77750805],
           [ 1.45361793, -0.48119892,  0.9954361 ,  0.36783024],
           [-1.23601035, -1.03730773,  0.98112443, -0.90393631],
           [-0.83958668,  0.04965843, -0.43122089,  0.60366836],
           [-1.42041273,  0.41266751, -0.9637889 ,  0.47148076],
           [ 0.38529206,  0.83155384, -0.78103554, -1.11777487],
           [ 0.74723321,  0.2154795 , -0.53567181, -0.23179069]])




```python
# Get dataframe column as NumPy presentation
import numpy as np
X = np.c_[df['A']]
X
```




    array([[-0.39668765],
           [ 0.5256545 ],
           [ 1.46105881],
           [ 2.06909247],
           [ 1.45361793],
           [-1.23601035],
           [-0.83958668],
           [-1.42041273],
           [ 0.38529206],
           [ 0.74723321]])




```python
# Shape of A are rows and columns of the dataframe
A.shape
```




    (10, 4)




```python
# Show a stat summary of the data
df.describe()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.776800</td>
      <td>0.823109</td>
      <td>0.156050</td>
      <td>0.289448</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.368299</td>
      <td>0.955949</td>
      <td>1.100099</td>
      <td>1.360076</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.704010</td>
      <td>-0.383277</td>
      <td>-1.479013</td>
      <td>-1.416606</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.862804</td>
      <td>-0.049376</td>
      <td>-0.681005</td>
      <td>-0.395471</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.528841</td>
      <td>0.883935</td>
      <td>0.124519</td>
      <td>0.092420</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.156078</td>
      <td>1.655364</td>
      <td>0.851772</td>
      <td>1.017481</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.526778</td>
      <td>2.062913</td>
      <td>1.709313</td>
      <td>3.173983</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Show a summary of our dataframe
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 10 entries, a to j
    Data columns (total 4 columns):
    A    10 non-null float64
    B    10 non-null float64
    C    10 non-null float64
    D    10 non-null float64
    dtypes: float64(4)
    memory usage: 720.0+ bytes
    


```python
# Transpose data
df.T
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
      <th>g</th>
      <th>h</th>
      <th>i</th>
      <th>j</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.260968</td>
      <td>-0.158592</td>
      <td>-3.704010</td>
      <td>-0.454166</td>
      <td>-2.583424</td>
      <td>-0.629530</td>
      <td>-0.940561</td>
      <td>0.526778</td>
      <td>0.518058</td>
      <td>-0.603516</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1.462966</td>
      <td>1.765306</td>
      <td>0.346566</td>
      <td>1.719496</td>
      <td>-0.319739</td>
      <td>1.421304</td>
      <td>-0.176530</td>
      <td>-0.383277</td>
      <td>2.062913</td>
      <td>0.332087</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.935928</td>
      <td>0.153076</td>
      <td>1.709313</td>
      <td>0.095962</td>
      <td>-0.212480</td>
      <td>-0.837180</td>
      <td>-1.086958</td>
      <td>0.599304</td>
      <td>1.682550</td>
      <td>-1.479013</td>
    </tr>
    <tr>
      <th>D</th>
      <td>-1.416606</td>
      <td>-0.099588</td>
      <td>3.173983</td>
      <td>-1.357128</td>
      <td>1.210537</td>
      <td>0.438314</td>
      <td>-0.494099</td>
      <td>0.267912</td>
      <td>-0.083072</td>
      <td>1.254228</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sort by values
df.sort_values(by='C', ascending=False)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>-3.704010</td>
      <td>0.346566</td>
      <td>1.709313</td>
      <td>3.173983</td>
    </tr>
    <tr>
      <th>i</th>
      <td>0.518058</td>
      <td>2.062913</td>
      <td>1.682550</td>
      <td>-0.083072</td>
    </tr>
    <tr>
      <th>a</th>
      <td>0.260968</td>
      <td>1.462966</td>
      <td>0.935928</td>
      <td>-1.416606</td>
    </tr>
    <tr>
      <th>h</th>
      <td>0.526778</td>
      <td>-0.383277</td>
      <td>0.599304</td>
      <td>0.267912</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.158592</td>
      <td>1.765306</td>
      <td>0.153076</td>
      <td>-0.099588</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-0.454166</td>
      <td>1.719496</td>
      <td>0.095962</td>
      <td>-1.357128</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-2.583424</td>
      <td>-0.319739</td>
      <td>-0.212480</td>
      <td>1.210537</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-0.629530</td>
      <td>1.421304</td>
      <td>-0.837180</td>
      <td>0.438314</td>
    </tr>
    <tr>
      <th>g</th>
      <td>-0.940561</td>
      <td>-0.176530</td>
      <td>-1.086958</td>
      <td>-0.494099</td>
    </tr>
    <tr>
      <th>j</th>
      <td>-0.603516</td>
      <td>0.332087</td>
      <td>-1.479013</td>
      <td>1.254228</td>
    </tr>
  </tbody>
</table>
</div>



### Selection


```python
# Initial Dataframe
df = pd.DataFrame(np.random.randn(10, 4), index=list('abcdefghij'),
             columns=list('ABCD'))
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.294648</td>
      <td>0.023458</td>
      <td>-0.735998</td>
      <td>-0.552704</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.030628</td>
      <td>1.069447</td>
      <td>0.483863</td>
      <td>0.723867</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.253603</td>
      <td>0.066511</td>
      <td>0.141998</td>
      <td>0.406838</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-0.807581</td>
      <td>-0.065567</td>
      <td>-0.399008</td>
      <td>0.321400</td>
    </tr>
    <tr>
      <th>e</th>
      <td>0.417448</td>
      <td>-0.397863</td>
      <td>1.852155</td>
      <td>2.455886</td>
    </tr>
    <tr>
      <th>f</th>
      <td>0.287584</td>
      <td>0.317224</td>
      <td>0.767366</td>
      <td>-0.053437</td>
    </tr>
    <tr>
      <th>g</th>
      <td>0.457348</td>
      <td>-1.253723</td>
      <td>2.869324</td>
      <td>-0.327216</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.268315</td>
      <td>-0.588351</td>
      <td>1.476119</td>
      <td>-0.224964</td>
    </tr>
    <tr>
      <th>i</th>
      <td>0.436500</td>
      <td>1.178807</td>
      <td>0.452377</td>
      <td>0.083320</td>
    </tr>
    <tr>
      <th>j</th>
      <td>-1.013040</td>
      <td>-1.230992</td>
      <td>-0.021071</td>
      <td>3.256197</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select a single column. Returns a Series
df['A']
```




    a   -0.294648
    b   -1.030628
    c    0.253603
    d   -0.807581
    e    0.417448
    f    0.287584
    g    0.457348
    h   -0.268315
    i    0.436500
    j   -1.013040
    Name: A, dtype: float64




```python
# Select a single column. Returns a Series
df.B
```




    a    0.023458
    b    1.069447
    c    0.066511
    d   -0.065567
    e   -0.397863
    f    0.317224
    g   -1.253723
    h   -0.588351
    i    1.178807
    j   -1.230992
    Name: B, dtype: float64




```python
# Select a slice of rows. Recall indexes start at 0
df[2:4]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>0.253603</td>
      <td>0.066511</td>
      <td>0.141998</td>
      <td>0.406838</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-0.807581</td>
      <td>-0.065567</td>
      <td>-0.399008</td>
      <td>0.321400</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select on multi-axis
df.loc[:, ['A', 'D']]
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
      <th>A</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.294648</td>
      <td>-0.552704</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.030628</td>
      <td>0.723867</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.253603</td>
      <td>0.406838</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-0.807581</td>
      <td>0.321400</td>
    </tr>
    <tr>
      <th>e</th>
      <td>0.417448</td>
      <td>2.455886</td>
    </tr>
    <tr>
      <th>f</th>
      <td>0.287584</td>
      <td>-0.053437</td>
    </tr>
    <tr>
      <th>g</th>
      <td>0.457348</td>
      <td>-0.327216</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.268315</td>
      <td>-0.224964</td>
    </tr>
    <tr>
      <th>i</th>
      <td>0.436500</td>
      <td>0.083320</td>
    </tr>
    <tr>
      <th>j</th>
      <td>-1.013040</td>
      <td>3.256197</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select a row on multi-axis. Returns a Series
df.loc['c', ['A', 'D']]
```




    A    0.253603
    D    0.406838
    Name: c, dtype: float64




```python
# Select on multi-axis
df.loc[['a', 'c'], ['A', 'D']]
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
      <th>A</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.294648</td>
      <td>-0.552704</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.253603</td>
      <td>0.406838</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get a scalar value
df.loc['c', 'C']
```




    0.14199778789540501




```python
# Get a scalar value using 'at'
df.at['c', 'C']
```




    0.14199778789540501




```python
# Set values on a dataframe
indices = list(['a', 'd', 'f'])
df.loc[indices, 'D'] = 100
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.294648</td>
      <td>0.023458</td>
      <td>-0.735998</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.030628</td>
      <td>1.069447</td>
      <td>0.483863</td>
      <td>0.723867</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.253603</td>
      <td>0.066511</td>
      <td>0.141998</td>
      <td>0.406838</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-0.807581</td>
      <td>-0.065567</td>
      <td>-0.399008</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>e</th>
      <td>0.417448</td>
      <td>-0.397863</td>
      <td>1.852155</td>
      <td>2.455886</td>
    </tr>
    <tr>
      <th>f</th>
      <td>0.287584</td>
      <td>0.317224</td>
      <td>0.767366</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>g</th>
      <td>0.457348</td>
      <td>-1.253723</td>
      <td>2.869324</td>
      <td>-0.327216</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.268315</td>
      <td>-0.588351</td>
      <td>1.476119</td>
      <td>-0.224964</td>
    </tr>
    <tr>
      <th>i</th>
      <td>0.436500</td>
      <td>1.178807</td>
      <td>0.452377</td>
      <td>0.083320</td>
    </tr>
    <tr>
      <th>j</th>
      <td>-1.013040</td>
      <td>-1.230992</td>
      <td>-0.021071</td>
      <td>3.256197</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select by position
df.iloc[2]
```




    A    0.253603
    B    0.066511
    C    0.141998
    D    0.406838
    Name: c, dtype: float64




```python
# Select like slicing
df.iloc[2:4,1:3]
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
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>0.066511</td>
      <td>0.141998</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-0.065567</td>
      <td>-0.399008</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select specific locations
df.iloc[[1, 3],[0, 2]]
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
      <th>A</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>b</th>
      <td>-1.030628</td>
      <td>0.483863</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-0.807581</td>
      <td>-0.399008</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Slice rows
df.iloc[4:7, :]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>e</th>
      <td>0.417448</td>
      <td>-0.397863</td>
      <td>1.852155</td>
      <td>2.455886</td>
    </tr>
    <tr>
      <th>f</th>
      <td>0.287584</td>
      <td>0.317224</td>
      <td>0.767366</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>g</th>
      <td>0.457348</td>
      <td>-1.253723</td>
      <td>2.869324</td>
      <td>-0.327216</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Slice columns
df.iloc[:, 2:4]
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
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.735998</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.483863</td>
      <td>0.723867</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.141998</td>
      <td>0.406838</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-0.399008</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>e</th>
      <td>1.852155</td>
      <td>2.455886</td>
    </tr>
    <tr>
      <th>f</th>
      <td>0.767366</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>g</th>
      <td>2.869324</td>
      <td>-0.327216</td>
    </tr>
    <tr>
      <th>h</th>
      <td>1.476119</td>
      <td>-0.224964</td>
    </tr>
    <tr>
      <th>i</th>
      <td>0.452377</td>
      <td>0.083320</td>
    </tr>
    <tr>
      <th>j</th>
      <td>-0.021071</td>
      <td>3.256197</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get a scalar value at specific position
df.iloc[2,3]
```




    0.40683806053761956




```python
# Get a scalar value at specific position using 'iat'
df.iat[2,3]
```




    0.40683806053761956




```python
# Boolean indexing

```

### Common features


```python
# Load dataset into a dataframe
import os
datapath = os.path.join("datasets", "islr", "")
auto = pd.read_csv(datapath + "dataset_filename.csv", 
                   delim_whitespace=True, na_values='?')

```


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
# Count values for a particular column
auto['year'].value_counts()
```




    73    40
    78    36
    76    34
    82    30
    75    30
    81    29
    80    29
    79    29
    70    29
    77    28
    72    28
    71    28
    74    27
    Name: year, dtype: int64




```python
# Generate a correlation matrix
auto.corr()
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
  </thead>
  <tbody>
    <tr>
      <th>mpg</th>
      <td>1.000000</td>
      <td>-0.776260</td>
      <td>-0.804443</td>
      <td>-0.778427</td>
      <td>-0.831739</td>
      <td>0.422297</td>
      <td>0.581469</td>
      <td>0.563698</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>-0.776260</td>
      <td>1.000000</td>
      <td>0.950920</td>
      <td>0.842983</td>
      <td>0.897017</td>
      <td>-0.504061</td>
      <td>-0.346717</td>
      <td>-0.564972</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>-0.804443</td>
      <td>0.950920</td>
      <td>1.000000</td>
      <td>0.897257</td>
      <td>0.933104</td>
      <td>-0.544162</td>
      <td>-0.369804</td>
      <td>-0.610664</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>-0.778427</td>
      <td>0.842983</td>
      <td>0.897257</td>
      <td>1.000000</td>
      <td>0.864538</td>
      <td>-0.689196</td>
      <td>-0.416361</td>
      <td>-0.455171</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>-0.831739</td>
      <td>0.897017</td>
      <td>0.933104</td>
      <td>0.864538</td>
      <td>1.000000</td>
      <td>-0.419502</td>
      <td>-0.307900</td>
      <td>-0.581265</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>0.422297</td>
      <td>-0.504061</td>
      <td>-0.544162</td>
      <td>-0.689196</td>
      <td>-0.419502</td>
      <td>1.000000</td>
      <td>0.282901</td>
      <td>0.210084</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.581469</td>
      <td>-0.346717</td>
      <td>-0.369804</td>
      <td>-0.416361</td>
      <td>-0.307900</td>
      <td>0.282901</td>
      <td>1.000000</td>
      <td>0.184314</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>0.563698</td>
      <td>-0.564972</td>
      <td>-0.610664</td>
      <td>-0.455171</td>
      <td>-0.581265</td>
      <td>0.210084</td>
      <td>0.184314</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Copy a dataframe
auto_copy = auto.copy()
```


```python
# Clean data with NaN Values for a column
auto_copy = auto_copy.dropna(subset=['year'])
```


```python
# Clean data with NaN Values for all dataframe
auto_copy = auto_copy.dropna()
```


```python
# Drop a column
auto_copy = auto_copy.drop('name', axis=1)
auto_copy.head()
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
    </tr>
  </tbody>
</table>
</div>




```python
# Make a list from dataframe column names
list(auto_copy)
```




    ['mpg',
     'cylinders',
     'displacement',
     'horsepower',
     'weight',
     'acceleration',
     'year',
     'origin']



### References
http://pandas.pydata.org/pandas-docs/stable/index.html
