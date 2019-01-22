---
layout: post
title:  "Getting Started with Octave"
date:   2019-01-21 19:20:34 -0500
categories: [Data science]
tags: [octave, introduction]
excerpt_separator: <!--excerpt-->
---
A brief introduction to Octave, a numerical tool that makes math much easier. 
<!--excerpt-->

## Basic Intro

### Basic Operations


```matlab
% Basic Operations
4 + 6 
8 - 2 
7*2
8/3
3^4
```

    ans =  10
    ans =  6
    ans =  14
    ans =  2.6667
    ans =  81
    

### Basic Logical Operations


```matlab
% Basic Logical Operations
4 == 5 % false
7 ~= 10 % true
1 && 0 % false
0 || 1 % true
```

    ans = 0
    ans = 1
    ans = 0
    ans = 1
    

### Variable Assignment


```matlab
% Variable Assignment
x = 10
y = 'Hello'
z = 10<=5  % false
```

    x =  10
    y = Hello
    z = 0
    

### Display variables


```matlab
% Display variables
p = pi
disp(p)
% Displaying with format
disp(sprintf('pi showing 2 decimals: %0.2f', p))
```

    p =  3.1416
     3.1416
    pi showing 2 decimals: 3.14
    

### Vector and Matrices


```matlab
% Vector and Matrices
A = [1 2; 3 4; 5 6] % 3x2 Matrix
r = [ 4 5 6] % row vector
v = [ 7; 8; 9] % column Vector
```

    A =
    
       1   2
       3   4
       5   6
    
    r =
    
       4   5   6
    
    v =
    
       7
       8
       9
    
    

### Vector and Matrices - Generators


```matlab
% Common Generators
a = 1:0.2:2  % 0.2 Stepwise
b = 1:10 % Asumme stepwise 1
c = ones(3,4) % Matrix of ones
d = zeros(2,3) % Matrix of zeros
e = rand(2,5) % Matrix of random values with uniformed distribution
w = randn(4,3) % Matrix of random values with normal distribution
I = eye(5) % Identity matrix of 5x5
```

    a =
    
        1.0000    1.2000    1.4000    1.6000    1.8000    2.0000
    
    b =
    
        1    2    3    4    5    6    7    8    9   10
    
    c =
    
       1   1   1   1
       1   1   1   1
       1   1   1   1
    
    d =
    
       0   0   0
       0   0   0
    
    e =
    
       0.438657   0.233235   0.811803   0.872061   0.418583
       0.012942   0.042095   0.316758   0.263557   0.171856
    
    w =
    
      -0.048263   0.349883   0.667025
      -1.094688   0.994961  -0.032410
      -0.449045  -0.158395  -0.722067
       0.297615   0.221675  -0.592378
    
    I =
    
    Diagonal Matrix
    
       1   0   0   0   0
       0   1   0   0   0
       0   0   1   0   0
       0   0   0   1   0
       0   0   0   0   1
    
    
