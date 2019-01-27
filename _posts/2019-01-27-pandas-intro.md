---
layout: post
title:  "Introduction to Pandas"
date:   2019-01-27 09:20:34 -0500
categories: [Data science]
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




    -0.44784696685415826




```python
# Access and Set values by index values
s['a'] = 0
s
```




    a    0.000000
    b    0.161149
    c    0.647253
    d    0.440717
    e   -0.447847
    dtype: float64




```python
# Query if index exists
'c' in s
```




    True



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




    array([[ 0.26096846,  1.46296573,  0.93592846, -1.41660609],
           [-0.15859237,  1.76530578,  0.15307612, -0.09958755],
           [-3.70400969,  0.34656611,  1.70931306,  3.17398254],
           [-0.45416638,  1.71949633,  0.09596205, -1.35712752],
           [-2.58342432, -0.31973853, -0.21248009,  1.21053678],
           [-0.62953017,  1.42130367, -0.8371803 ,  0.43831372],
           [-0.94056138, -0.17653028, -1.08695849, -0.49409931],
           [ 0.52677801, -0.38327714,  0.59930409,  0.2679122 ],
           [ 0.51805825,  2.06291289,  1.68255019, -0.08307222],
           [-0.60351569,  0.33208733, -1.47901262,  1.2542283 ]])




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
      <td>-0.602492</td>
      <td>-0.547432</td>
      <td>0.185330</td>
      <td>0.123702</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.433955</td>
      <td>0.474081</td>
      <td>0.537173</td>
      <td>-0.990728</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.371477</td>
      <td>-0.535911</td>
      <td>-0.682828</td>
      <td>-0.640206</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1.625613</td>
      <td>0.162837</td>
      <td>0.429441</td>
      <td>-1.244517</td>
    </tr>
    <tr>
      <th>e</th>
      <td>0.189914</td>
      <td>1.560723</td>
      <td>-1.631715</td>
      <td>-2.057348</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-0.254064</td>
      <td>1.154083</td>
      <td>0.285357</td>
      <td>0.047092</td>
    </tr>
    <tr>
      <th>g</th>
      <td>1.512219</td>
      <td>0.154066</td>
      <td>1.415969</td>
      <td>1.290783</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.504876</td>
      <td>1.576506</td>
      <td>-1.292428</td>
      <td>-1.065340</td>
    </tr>
    <tr>
      <th>i</th>
      <td>-0.504846</td>
      <td>-0.524116</td>
      <td>-0.566280</td>
      <td>1.037355</td>
    </tr>
    <tr>
      <th>j</th>
      <td>0.523905</td>
      <td>0.291844</td>
      <td>0.103938</td>
      <td>0.478190</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select a single column. Returns a Series
df['A']
```




    a   -0.602492
    b   -0.433955
    c    0.371477
    d    1.625613
    e    0.189914
    f   -0.254064
    g    1.512219
    h   -0.504876
    i   -0.504846
    j    0.523905
    Name: A, dtype: float64




```python
# Select a single column. Returns a Series
df.B
```




    a   -0.547432
    b    0.474081
    c   -0.535911
    d    0.162837
    e    1.560723
    f    1.154083
    g    0.154066
    h    1.576506
    i   -0.524116
    j    0.291844
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
      <td>0.371477</td>
      <td>-0.535911</td>
      <td>-0.682828</td>
      <td>-0.640206</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1.625613</td>
      <td>0.162837</td>
      <td>0.429441</td>
      <td>-1.244517</td>
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
      <td>-0.602492</td>
      <td>0.123702</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.433955</td>
      <td>-0.990728</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.371477</td>
      <td>-0.640206</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1.625613</td>
      <td>-1.244517</td>
    </tr>
    <tr>
      <th>e</th>
      <td>0.189914</td>
      <td>-2.057348</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-0.254064</td>
      <td>0.047092</td>
    </tr>
    <tr>
      <th>g</th>
      <td>1.512219</td>
      <td>1.290783</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.504876</td>
      <td>-1.065340</td>
    </tr>
    <tr>
      <th>i</th>
      <td>-0.504846</td>
      <td>1.037355</td>
    </tr>
    <tr>
      <th>j</th>
      <td>0.523905</td>
      <td>0.478190</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select a row on multi-axis. Returns a Series
df.loc['c', ['A', 'D']]
```




    A    0.371477
    D   -0.640206
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
      <td>-0.602492</td>
      <td>0.123702</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.371477</td>
      <td>-0.640206</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get a scalar value
df.loc['c', 'C']
```




    -0.6828276144125267




```python
# Get a scalar value using 'at'
df.at['c', 'C']
```




    -0.6828276144125267




```python
# Select by position
df.iloc[2]
```




    A    0.371477
    B   -0.535911
    C   -0.682828
    D   -0.640206
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
      <td>-0.535911</td>
      <td>-0.682828</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.162837</td>
      <td>0.429441</td>
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
      <td>-0.433955</td>
      <td>0.537173</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1.625613</td>
      <td>0.429441</td>
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
      <td>0.189914</td>
      <td>1.560723</td>
      <td>-1.631715</td>
      <td>-2.057348</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-0.254064</td>
      <td>1.154083</td>
      <td>0.285357</td>
      <td>0.047092</td>
    </tr>
    <tr>
      <th>g</th>
      <td>1.512219</td>
      <td>0.154066</td>
      <td>1.415969</td>
      <td>1.290783</td>
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
      <td>0.185330</td>
      <td>0.123702</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.537173</td>
      <td>-0.990728</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-0.682828</td>
      <td>-0.640206</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.429441</td>
      <td>-1.244517</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-1.631715</td>
      <td>-2.057348</td>
    </tr>
    <tr>
      <th>f</th>
      <td>0.285357</td>
      <td>0.047092</td>
    </tr>
    <tr>
      <th>g</th>
      <td>1.415969</td>
      <td>1.290783</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-1.292428</td>
      <td>-1.065340</td>
    </tr>
    <tr>
      <th>i</th>
      <td>-0.566280</td>
      <td>1.037355</td>
    </tr>
    <tr>
      <th>j</th>
      <td>0.103938</td>
      <td>0.478190</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get a scalar value at specific position
df.iloc[2,3]
```




    -0.6402062109728908




```python
# Get a scalar value at specific position using 'iat'
df.iat[2,3]
```




    -0.6402062109728908




```python
# Boolean indexing

```

### References
http://pandas.pydata.org/pandas-docs/stable/index.html
