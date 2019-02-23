---
layout: post
title:  "Introduction to NumPy"
date:   2019-01-26 17:20:34 -0500
categories: [Python]
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


```python
# Create an array of permutation
np.random.permutation(10)
```




    array([8, 9, 3, 5, 4, 0, 1, 7, 6, 2])




```python
# Create an array of random numbers
np.random.rand(3,4)
```




    array([[0.71794374, 0.96419672, 0.30552228, 0.91227883],
           [0.30435182, 0.48216834, 0.66421105, 0.52341418],
           [0.61700862, 0.13274624, 0.68705621, 0.85937783]])




```python
# Create an array of random (normal distribution)
np.random.randn(3,4)
```




    array([[ 0.73470783,  0.12051316, -0.60581287,  2.45788469],
           [ 0.72720299, -0.40864697,  1.50380995,  0.0881882 ],
           [-0.48701366, -1.0951752 ,  0.97543857, -0.72006671]])



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
    

## Indexing, Slicing and Iterating

### One Dimensional


```python
a = np.arange(10)**2
a
```




    array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81], dtype=int32)




```python
a[4]
```




    16




```python
a[2:5]
```




    array([ 4,  9, 16], dtype=int32)




```python
# from start to position 9, exclusive, set every three elements
a[:9:3] = 100  # equivalent to a[0:9:3] 
a
```




    array([100,   1,   4, 100,  16,  25, 100,  49,  64,  81], dtype=int32)



### Two Dimensional


```python
b = (np.arange(12)**2).reshape(3,4)
b
```




    array([[  0,   1,   4,   9],
           [ 16,  25,  36,  49],
           [ 64,  81, 100, 121]], dtype=int32)




```python
b[2,3]
```




    121




```python
b[0:3, 1]
```




    array([ 1, 25, 81], dtype=int32)




```python
b[ : ,1]
```




    array([ 1, 25, 81], dtype=int32)




```python
b[1:3, : ]
```




    array([[ 16,  25,  36,  49],
           [ 64,  81, 100, 121]], dtype=int32)



### Three Dimensional


```python
c = (np.arange(12)).reshape(2,2,3)
c
```




    array([[[ 0,  1,  2],
            [ 3,  4,  5]],
    
           [[ 6,  7,  8],
            [ 9, 10, 11]]])




```python
c[1,...]
```




    array([[ 6,  7,  8],
           [ 9, 10, 11]])




```python
c[...,2]
```




    array([[ 2,  5],
           [ 8, 11]])



### Iterating


```python
# Iterate over the first axis
for row in b:
    print(row)
```

    [0 1 4 9]
    [16 25 36 49]
    [ 64  81 100 121]
    


```python
# if one wants to perform an operation on each element
for element in b.flat:
    print(element)
```

    0
    1
    4
    9
    16
    25
    36
    49
    64
    81
    100
    121
    

### Common Functions


```python
# Translate slice objects to concatenation along the second axis.
np.c_[np.array([1,2,3]), np.array([4,5,6])]
```




    array([[1, 4],
           [2, 5],
           [3, 6]])




```python
# Keep a same seed in different executions
np.random.seed(1)
```

### References
http://www.numpy.org/
