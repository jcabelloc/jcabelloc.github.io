---
layout: post
title:  "Getting Started with Python 3"
date:   2019-01-24 22:20:34 -0500
categories: [Python]
tags: [python, introduction]
---


This is a brief introduction to Python 3, a scripting language widely used in data science and machine learning.

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



### Comparison Operators


```python
# Comparison
1 > 5
```




    False




```python
# Comparison
1 < 8
```




    True




```python
# Comparison
1 <= 1
```




    True




```python
# Comparison
8 == 8
```




    True




```python
# Comparison
'by' == 'goodbye'
```




    False



### Logical Operators


```python
# Operator: and
a = 10
a > 5 and a < 20
```




    True




```python
# Operator: or
a = 10
a >= 5 or a < 8
```




    True




```python
# Operator: not
a = 10
not (a < 5)
```




    True



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
Lists might contain items of different types, but usually the items all have the same type. Lists are mutable sequences. 


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




```python
# List from a String
letters = list('PYTHON')
letters
```




    ['P', 'Y', 'T', 'H', 'O', 'N']




```python
# Check if element is in a list
letters = list('PYTHON')
'P' in letters
```




    True




```python
# Get index of a element in a list
fruits = list(['orange', 'apple', 'banana', 'grapes'])
fruits.index('banana')
```




    2



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
    


```python
# Conditional assignment
grade = 9
result = 'A' if grade > 9 else 'B'
result
```




    'B'



### For Statements
Pythonâ€™s for statement iterates over the items of any sequence (a list or a string)


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
    


```python
for i in range(5):
    print(i**2)
```

    0
    1
    4
    9
    16
    

### While Statement


```python
i = 1
while i <= 3 :
    print('i is: {}'.format(i))
    i = i + 1
```

    i is: 1
    i is: 2
    i is: 3
    

### pass Statements
"The pass statement does nothing. It can be used when a statement is required syntactically but the program requires no action"


```python
for i in [1,2,3]:
    pass  # Does nothing
```

### The range() Function
The range type represents an immutable sequence of numbers and is commonly used for looping a specific number of times in for loops.


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



### Functions


```python
# The keyword def introduces a function definition.
def square(n): # Calculates the square of a number n
    return n ** 2
```


```python
# Calling our square function
square(3)
```




    9




```python
# Assigning the function to a variable and calling that
sq = square
sq(4)
```




    16




```python
# Default argument value
def repeatPrint(message, times=5, finalMsg='Bye'):
    i = 1
    while i <= times:
        i += 1
        print(message)
    print(finalMsg)

    
```


```python
# Calling the function with all arguments
repeatPrint('Hello', 2, 'Good bye')
```

    Hello
    Hello
    Good bye
    


```python
# Calling the function without optinal argument
repeatPrint('Hello')
```

    Hello
    Hello
    Hello
    Hello
    Hello
    Bye
    


```python
# Using keyword Arguments
repeatPrint(times=3, message='Hi', finalMsg='HoHoHo')
```

    Hi
    Hi
    Hi
    HoHoHo
    


```python
# Keywords and arguments
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])
# 
# Calling the defined function
cheeseshop("Limburger", "It's very runny, sir.",
           "It's really very, VERY runny, sir.",
           shopkeeper="Michael Palin",
           client="John Cleese",
           sketch="Cheese Shop Sketch")
```

    -- Do you have any Limburger ?
    -- I'm sorry, we're all out of Limburger
    It's very runny, sir.
    It's really very, VERY runny, sir.
    ----------------------------------------
    shopkeeper : Michael Palin
    client : John Cleese
    sketch : Cheese Shop Sketch
    


```python
# Unpacking Argument Lists
list(range(3, 6))  # normal call with separate arguments
args = [3, 6]      # List 
list(range(*args))  # call with arguments unpacked from a list
```




    [3, 4, 5]




```python
# Documentation of Functions
def myFunction():
    """Do nothing, but document it.
    
    No, really, it doesn't do anything.
        """
    pass
#
print(myFunction.__doc__)

```

    Do nothing, but document it.
        
        No, really, it doesn't do anything.
            
    

### Lambda Expressions


```python
# Define the lambda expression
def any_exp(n):
    return lambda x: x**n

# Call the lambda expression
cubic = any_exp(3)
cubic(4)
```




    64



### Tuples
"Tuples are immutable sequences, typically used to store collections of heterogeneous data". "A tuple consists of a number of values separated by commas"


```python
# Defining a tuple with heterogenous data
t = 12345, 54321, 'hello!'
t[0]  # 12345
t
```




    (12345, 54321, 'hello!')




```python
# Tuples may be nested:
u = t, (1, 2, 3, 4, 5)
u
# Tuples are immutable:
# t[0] = 100 # yields error
```




    ((12345, 54321, 'hello!'), (1, 2, 3, 4, 5))




```python
# Tuples enclosed in parentheses
customer = (1234, 'Texas', 'Male')
customer
```




    (1234, 'Texas', 'Male')




```python
# zip() used to generate an iterator of tuples
# returns a zip object that is an iterator of tuples
a = [1, 2, 3]
b = [4, 5, 6]
c = zip(a, b)
for t in c:
    print(t)
```

    (1, 4)
    (2, 5)
    (3, 6)
    

### Sets
"Python also includes a data type for sets. A set is an unordered collection with no duplicate elements. Basic uses include membership testing and eliminating duplicate entries"


```python
# Duplicates are removed
countries = {'Peru', 'Mexico', 'Chile', 'Peru', 'Canada', 'Chile'}
countries
```




    {'Canada', 'Chile', 'Mexico', 'Peru'}




```python
# Membershing validation
'Peru' in countries
```




    True




```python
# set() can be used to create sets
set_let = set('tennessee')
set_let
```




    {'e', 'n', 's', 't'}



### Dictionaries
"It is best to think of a dictionary as a set of key: value pairs, with the requirement that the keys are unique (within one dictionary)"


```python
# Defining a dictionary
tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127
tel
```




    {'jack': 4098, 'sape': 4139, 'guido': 4127}




```python
# Accesing a value
tel['jack']
```




    4098




```python
# Delete a key-value
del tel['sape']
tel
```




    {'jack': 4098, 'guido': 4127}




```python
## Adding a key-value
tel['irv'] = 4127
tel
```




    {'jack': 4098, 'guido': 4127, 'irv': 4127}




```python
# List the keys
list(tel)
```




    ['jack', 'guido', 'irv']




```python
# Sort the list of keys
sorted(tel)
```




    ['guido', 'irv', 'jack']




```python
# Validate the existance of a key
'guido' in tel
```




    True




```python
# Construct a dictionary from key-pair value
dict(sape=4139, guido=4127, jack=4098)
```




    {'sape': 4139, 'guido': 4127, 'jack': 4098}



### Other common functions


```python
# Print supports multiparameters
pi = 3.1416
print('The value of pi is:', pi)
```

    The value of pi is: 3.1416
    


```python
# Print with format
num = '123';
name = 'Jhon'
print('My number is: {a}, and my name is: {b}'.format(a=num,b=name))
```

    My number is: 123, and my name is: Jhon
    


```python
# Type of an object
l = list('UTIL');
type(l)
```




    list




```python
# Help/Documentation
len?
```


```python
# Set a O.S. directory path
import os
datapath = os.path.join("datasets", "islr", "")
```


```python
# Capture the enlapsed time of a task
import time
tic = time.time();
time.sleep(2)
toc = time.time();
print('The enlapsed time was: ' + str(toc - tic) + " seg")
```

    The enlapsed time was: 2.00048565864563 seg
    

### References:
https://docs.python.org/3/tutorial/
