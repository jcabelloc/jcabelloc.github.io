---
layout: post
title:  "Introduction to NumPy"
date:   2019-01-26 17:20:34 -0500
categories: [Data science]
tags: [python, numpy, introduction]
---

A brief introduction to NumPy

"NumPy is the fundamental package for scientific computing with Python. It contains among other things"

### The basics


```python
# Import NumPy
import numpy as np
```


```python
# Create an array from a List
l = [1, 2, 3, 4, 5, 6]
a = np.array(l)
a
```




    array([1, 2, 3, 4, 5, 6])




```python
# Reshape an Array
b = a.reshape(2,3)
b
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
# Query the shape
b.shape
```




    (2, 3)




```python
# Reshape an Array to 3 dimensions
a = np.array(range(1,25))
c = a.reshape(2,4,3)
c
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6],
            [ 7,  8,  9],
            [10, 11, 12]],
    
           [[13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24]]])




```python
# Query the number of axes(dimensions) of the array
c.ndim
```




    3




```python
# Query the total number of elements of an array
c.size
```




    24




```python
# Query the data type of the elements in the array
c.dtype.name
```




    'int32'




```python
# Division might alter the data type
c = c / 5
c.dtype.name
```




    'float64'



### Array Creacion


```python
# Create an array from a list
a = np.array([1, 2, 3, 4, 5, 6])
a
```




    array([1, 2, 3, 4, 5, 6])




```python
# Create an array from a sequence
b = np.array((4, 5, 6, 7))
b
```




    array([4, 5, 6, 7])




```python
# 2D arrays from sequences
np.array([(1, 2, 3), (4, 5, 6)])
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
# The function zeros creates an array full of zeros
np.zeros((3,4))
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
# The function ones creates an array full of ones
np.ones((3,2))
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.]])




```python
# Create an Array as a sequence of n elements
np.arange(6) 
```




    array([0, 1, 2, 3, 4, 5])




```python
# Create an Array as a sequence of numbers
np.arange(2, 30, 4)  # 4 is the step
```




    array([ 2,  6, 10, 14, 18, 22, 26])




```python
# arange another example
np.arange(0, 1, 0.1) # 0.1 is the step
```




    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])




```python
# Use linspace when we have the number of elements
np.linspace(0, 5, 11)
```




    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])




```python
# useful to evaluate function at lots of points
x = np.linspace( 0, 2*3.1416, 100)
f = np.sin(x)
```

### Print an Array


```python
# Print an array
a = np.arange(12).reshape(3,4)
print(a)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    


```python
# Print a 3D array
b = np.arange(24).reshape(3,2,4)
print(b)
```

    [[[ 0  1  2  3]
      [ 4  5  6  7]]
    
     [[ 8  9 10 11]
      [12 13 14 15]]
    
     [[16 17 18 19]
      [20 21 22 23]]]
    

### Array Basic Operations


```python
# Scalar addition
a = np.array([10, 20, 30, 40])
a + 5
```




    array([15, 25, 35, 45])




```python
# Scalar multiplication
a * 2
```




    array([20, 40, 60, 80])




```python
# Aritmentic operations are elementwise. There must exist dimension agreement.
# Array array substraction.
b = np.array([10, 10, 10, 10])
a - b
```




    array([ 0, 10, 20, 30])




```python
# Elementwise multiplication
a * b
```




    array([100, 200, 300, 400])




```python
# Elementwise division
a / b
```




    array([1., 2., 3., 4.])




```python
# Elementwise square
a ** 2
```




    array([ 100,  400,  900, 1600], dtype=int32)




```python
# Condition Evaluation is element by element
a <= 20
```




    array([ True,  True, False, False])




```python
# Matrix Multiplication with the @ operator
A = np.array([[1, 2], [2, 1]])
print('A:\n', A)
B = np.array([[2, 3], [3, 2]])
print('B\n', B)
C = A@B
print('AxB = \n', C)
```

    A:
     [[1 2]
     [2 1]]
    B
     [[2 3]
     [3 2]]
    AxB = 
     [[8 7]
     [7 8]]
    


```python
# Another way for Matrix Multiplication
A.dot(B)
```




    array([[8, 7],
           [7, 8]])




```python
# Min, max and sum for Arrays 1D
a = np.arange(6)
print(a)
print('Min: ', a.min())
print('Max: ', a.max())
print('Sum: ', a.sum())

```

    [0 1 2 3 4 5]
    Min:  0
    Max:  5
    Sum:  15
    


```python
# Min, max and sum for Matrices
A = np.arange(12).reshape(3,4)
print(A, '\n')
print('Min: ', A.min(axis=1))  # Min of each row
print('Max: ', A.max(axis=0))  # Max of each column
print('Sum: ', A.sum(axis=1))  # Sum of each row
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]] 
    
    Min:  [0 4 8]
    Max:  [ 8  9 10 11]
    Sum:  [ 6 22 38]
    


```python
# Mathematical Functions are elementwise
a = np.arange(5)
print('a: ', a)
print('power of e: ', np.exp(a))
print('sqrt: ', np.sqrt(a))

```

    a:  [0 1 2 3 4]
    power of e:  [ 1.          2.71828183  7.3890561  20.08553692 54.59815003]
    sqrt:  [0.         1.         1.41421356 1.73205081 2.        ]
    

### References
http://www.numpy.org/
