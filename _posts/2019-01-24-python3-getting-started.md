---
layout: post
title:  "Getting Started with Python 3"
date:   2019-01-24 22:20:34 -0500
categories: [Data science]
tags: [python, introduction]
---



This is a brief introduction to Python 3, a scripting language widely used in data science. 

### Basic Operations


```python
# Order of operators matters
4 * 5 - 12
```




    8




```python
# Simple Division returns a float
(3 + 9) / 4 
```




    3.0




```python
# Floor Division disregards the fractional part
13 // 5
```




    2




```python
# Geeting the remainder of a division
13 % 5
```




    3




```python
# Power of a number
2 ** 4
```




    16




```python
# Variable assigment
a = 3
b = 4
h = (a**2 + b**2)**(1/2)
h
```




    5.0



### Strings


```python
# String handling
str1 = 'Hello'
str2 = " variable 'x'"
str3 = " and variable \'y\'"
str4 = str1 + str2 + str3
str4
```




    "Hello variable 'x' and variable 'y'"




```python
# \n causes a new line
str = 'First line.\nSecond line.'
print(str)
```

    First line.
    Second line.
    


```python
# Indexing strings
word = "Python"
word[0]
```




    'P'




```python
# Slicing Strings: Characters from position 1 to position 2, position 3 is excluded
word[1:3] 
```




    'yt'




```python
# Characters from position 2 to the end
word[2:] 
```




    'thon'




```python
# Length of a string
len(word)
```




    6



### Lists
Lists might contain items of different types, but usually the items all have the same type.


```python
# List of Integers
primes = [1, 2, 3, 5, 7]
primes
```




    [1, 2, 3, 5, 7]




```python
# Indexing returns an element
primes[-1]
```




    7




```python
# Slicing returns a new list
primes[:3]
```




    [1, 2, 3]




```python
# Concatenation
primes + [11,13,17]
```




    [1, 2, 3, 5, 7, 11, 13, 17]




```python
# Slice assignation
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
letters[2:5] = []
letters
```




    ['a', 'b', 'f', 'g']




```python
# Length of a List
len(letters)
```




    4



### Print


```python
# Print supports multiparameters
pi = 3.1416
print('The value of pi is:', pi)
```

    The value of pi is: 3.1416
    

### IF Statements


```python
x = int(input("Please enter an integer: "))
# if is the minimal required statement
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')
```

    Please enter an integer: -10
    Negative changed to zero
    

### For Statements
Python’s for statement iterates over the items of any sequence (a list or a string)


```python
# Obtain the length of some strings:
words = ['Python', 'Java', 'Pascal']
# use 'for' to iterate a list
for w in words:
    print(w, '->', len(w))
```

    Python -> 6
    Java -> 4
    Pascal -> 6
    

### The range() Function


```python
# range() defines an iterable, in this case of 4 values
for i in range(4):
    print(i)
```

    0
    1
    2
    3
    


```python
# range() can start at another number
for j in range(5,10):
    print(j)
```

    5
    6
    7
    8
    9
    


```python
# range() returns an object(iterable) which returns the successive items of 
# the desired sequence when you iterate over it
r = range(0, 10, 3)
r
```




    range(0, 10, 3)




```python
# The function list() creates lists from iterables
list(r)
```




    [0, 3, 6, 9]



### References:
https://docs.python.org/3/tutorial/
