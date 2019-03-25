---
layout: post
title:  "Introduction to NumPy"
date:   2019-01-26 17:20:34 -0500
categories: [Python]
tags: [python, numpy, introduction]
---


A brief introduction to NumPy

"NumPy is the fundamental package for scientific computing with Python. It contains among other things"

Last Update: 3/25/2019

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
# Reshape an array as a matrix of one row
a = np.array([1, 2, 3, 4, 5, 6])
a.reshape(1, -1), a.reshape(1, -1).shape
```




    (array([[1, 2, 3, 4, 5, 6]]), (1, 6))




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




    array([4, 2, 0, 9, 1, 8, 7, 6, 5, 3])




```python
# Create an array of random numbers
np.random.rand(3,4)
```




    array([[0.98861615, 0.57974522, 0.38014117, 0.55094822],
           [0.74533443, 0.66923289, 0.26491956, 0.06633483],
           [0.3700842 , 0.62971751, 0.21017401, 0.75275555]])




```python
# Create an array of random (normal distribution)
np.random.randn(3,4)
```




    array([[-0.09387704, -0.16977402, -0.54114463,  0.53794761],
           [ 0.39128265,  2.21191487, -0.16224463,  0.29117816],
           [ 0.10806266, -0.19953292,  0.2328323 ,  0.15539326]])




```python
# Create an array of random integers
np.random.randint(0, 10, (3, 4))
```




    array([[3, 8, 8, 0],
           [6, 7, 9, 5],
           [4, 9, 5, 2]])




```python
# Create an identity matrix
np.identity(3)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python
# Return a 2-D array with ones on the diagonal and zeros elsewhere.
np.eye(3, 4)
```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.]])



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
# Use of dot for Matrix Multiplication
A.dot(B)
```




    array([[8, 7],
           [7, 8]])




```python
# Use of dot for Matrix Multiplication
np.dot(A, B)
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
# Sum rows of a matrix keeping dimensions
A.sum(axis=1, keepdims=True)
```




    array([[ 6],
           [22],
           [38]])




```python
# Compare two arrays and returns a new array containing the element-wise maxima
print(A, '\n')
np.maximum(A, 5)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]] 
    
    




    array([[ 5,  5,  5,  5],
           [ 5,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
# Mathematical Functions are elementwise
a = np.array([1, 2, 4])
print('a: ', a)
print('power of e: ', np.exp(a))
print('sqrt: ', np.sqrt(a))
print('log: ', np.log(a))
print('abs: ', np.abs(a))

```

    a:  [1 2 4]
    power of e:  [ 2.71828183  7.3890561  54.59815003]
    sqrt:  [1.         1.41421356 2.        ]
    log:  [0.         0.69314718 1.38629436]
    abs:  [1 2 4]
    


```python
# Return a contiguous flattened array.
A = np.arange(12).reshape(3,4)
print(A, '\n')
A.ravel()
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]] 
    
    




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])




```python
# Remove single-dimensional entries from the shape of an array.
A = np.arange(12).reshape(1,12,1)
print(A, '\n')
np.squeeze(A)
```

    [[[ 0]
      [ 1]
      [ 2]
      [ 3]
      [ 4]
      [ 5]
      [ 6]
      [ 7]
      [ 8]
      [ 9]
      [10]
      [11]]] 
    
    




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])



## Matrix Operations


```python
# Transpose of a Matrix
A = np.arange(12).reshape(3,4)
print('A: ', A)
A.transpose()
```

    A:  [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    




    array([[ 0,  4,  8],
           [ 1,  5,  9],
           [ 2,  6, 10],
           [ 3,  7, 11]])




```python
# A transpose short way
A.T
```




    array([[ 0,  4,  8],
           [ 1,  5,  9],
           [ 2,  6, 10],
           [ 3,  7, 11]])




```python
# Compute the inverse of a Matrix
A = np.random.randint(0, 9, (2,2))
print('A: ', A)
Ainv = np.linalg.inv(A)
print('Ainv: ', Ainv)
```

    A:  [[5 6]
     [6 8]]
    Ainv:  [[ 2.   -1.5 ]
     [-1.5   1.25]]
    


```python
# Matrix Multiplication of A and Inverse of A returns a identity matrix
np.dot(A, Ainv)
```




    array([[1.0000000e+00, 8.8817842e-16],
           [0.0000000e+00, 1.0000000e+00]])



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


```python
# Obtain the index of the max value in the array
np.argmax(np.array([4, 0, 9,6]))
```




    2




```python
# Create an assertion condition to validate an array shape
A = np.arange(12).reshape(3,4)
print('A: ', A)
assert(A.shape == (3,4))
print('\nThis line is printed if the previuos assertion condition is True')
```

    A:  [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    
    This line is printed if the previuos assertion condition is True
    

### References
http://www.numpy.org/
