---
layout: post
title:  "Getting Started with octave"
date:   2019-01-21 19:20:34 -0500
categories: [Tools]
tags: [octave, introduction]
---


A brief introduction to Octave, a numerical tool that makes math much easier. 

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
    
    


```matlab
% Common Generators
a = 1:0.2:2  % Generate a vector with 0.2 Stepwise
b = 1:10 % Asumme stepwise 1
C = ones(3,4) % Matrix of ones
D = zeros(2,3) % Matrix of zeros
E = rand(2,5) % Matrix of random values with uniformed distribution
W = randn(4,3) % Matrix of random values with normal distribution
I = eye(5) % Identity matrix of 5x5
```

    a =
    
        1.0000    1.2000    1.4000    1.6000    1.8000    2.0000
    
    b =
    
        1    2    3    4    5    6    7    8    9   10
    
    C =
    
       1   1   1   1
       1   1   1   1
       1   1   1   1
    
    D =
    
       0   0   0
       0   0   0
    
    E =
    
       0.3139465   0.5320929   0.0065746   0.2191232   0.3255126
       0.6983155   0.1484724   0.7644968   0.3556319   0.0315069
    
    W =
    
      -0.713562   0.159652  -1.440057
       0.432062   1.298259   1.869419
      -0.971604   0.832845   0.680742
      -0.665553  -0.606924  -0.054982
    
    I =
    
    Diagonal Matrix
    
       1   0   0   0   0
       0   1   0   0   0
       0   0   1   0   0
       0   0   0   1   0
       0   0   0   0   1
    
    


```matlab
% Dimensions
A
size(A) % [(Number of rows) (Number of columms)]
size(A,1) % Number of rows
size(A,2) % Number of columns
length(A) % length of the longest dimension
```

    A =
    
       1   2
       3   4
       5   6
    
    ans =
    
       3   2
    
    ans =  3
    ans =  2
    ans =  3
    


```matlab
% Indexing
A = magic(4) # Returns a 4x4 magic matrix
A(3,:) % Get the third row
A(:,4) % Get the fourth column as a vector
A([1 4],:) % Get the 1st and 4th row
A(:) % Select all elements as a column vector
```

    A =
    
       16    2    3   13
        5   11   10    8
        9    7    6   12
        4   14   15    1
    
    ans =
    
        9    7    6   12
    
    ans =
    
       13
        8
       12
        1
    
    ans =
    
       16    2    3   13
        4   14   15    1
    
    ans =
    
       16
        5
        9
        4
        2
       11
        7
       14
        3
       10
        6
       15
       13
        8
       12
        1
    
    


```matlab
% Joining Data
A = [1 1; 2 2; 3 3]
B = [4 4; 5 5; 6 6] % same dims as A
C = [A B]  % concatenating A and B along rows
C = [A, B] % concatenating A and B along rows
C = [A; B] % Concatenating A and B along columns
```

    A =
    
       1   1
       2   2
       3   3
    
    B =
    
       4   4
       5   5
       6   6
    
    C =
    
       1   1   4   4
       2   2   5   5
       3   3   6   6
    
    C =
    
       1   1   4   4
       2   2   5   5
       3   3   6   6
    
    C =
    
       1   1
       2   2
       3   3
       4   4
       5   5
       6   6
    
    

### Calculations on Matrices


```matlab
% initialize variables
A = [1 1;2 2;3 3]
B = [4 4;5 5;6 6]
C = [1 1;2 2]
v = [1;2;3]
```

    A =
    
       1   1
       2   2
       3   3
    
    B =
    
       4   4
       5   5
       6   6
    
    C =
    
       1   1
       2   2
    
    v =
    
       1
       2
       3
    
    


```matlab
% Matrix multiplication
A * C  % matrix multiplication
A .* B % element-wise multiplication
% A .* C  or A * B gives error - wrong dimensions
```

    ans =
    
       3   3
       6   6
       9   9
    
    ans =
    
        4    4
       10   10
       18   18
    
    


```matlab
% Explicit and implicit element-wise operations
A .^ 2 % element-wise square of each element in A
1./B   % element-wise reciprocal
log(v)  % functions like this operate element-wise on vecs or matrices 
exp(v)
abs(v)
v + 1
```

    ans =
    
       1   1
       4   4
       9   9
    
    ans =
    
       0.25000   0.25000
       0.20000   0.20000
       0.16667   0.16667
    
    ans =
    
       0.00000
       0.69315
       1.09861
    
    ans =
    
        2.7183
        7.3891
       20.0855
    
    ans =
    
       1
       2
       3
    
    ans =
    
       2
       3
       4
    
    


```matlab
v = [2 -10 3] % Vector
max(v) % Returns the max element of the vector
A = magic(3) % Matrix 3x3
max(A) % Returns the max element of each column
[val, ind] = max(A) % Returns the values and indices of those values
```

    v =
    
        2  -10    3
    
    ans =  3
    A =
    
       8   1   6
       3   5   7
       4   9   2
    
    ans =
    
       8   9   7
    
    val =
    
       8   9   7
    
    ind =
    
       1   3   2
    
    


```matlab
A <= 5 % Returns for each element 1(True) or 0(False) based on the condition
[r c] = find(A<=5); % gets row and column of elements matching the condition
[r c]
```

    ans =
    
      0  1  0
      1  1  0
      1  0  1
    
    ans =
    
       2   1
       3   1
       1   2
       2   2
       3   3
    
    


```matlab
A = [1 1; 3 3; 5 5]
sum(A) % Sum along the columns
sum(A,1) % Sum along the columns
sum(A,2) % Sum along the rows
prod(A) % Product along the columns
```

    A =
    
       1   1
       3   3
       5   5
    
    ans =
    
       9   9
    
    ans =
    
       9   9
    
    ans =
    
        2
        6
       10
    
    ans =
    
       15   15
    
    


```matlab
% Matrix inverse(pseudo-inverse)
A = magic(3)
Ai = pinv(A)
A * Ai
```

    A =
    
       8   1   6
       3   5   7
       4   9   2
    
    Ai =
    
       0.147222  -0.144444   0.063889
      -0.061111   0.022222   0.105556
      -0.019444   0.188889  -0.102778
    
    ans =
    
       1.0000e+00  -1.2212e-14   6.3283e-15
       5.5511e-17   1.0000e+00  -2.2204e-16
      -5.9952e-15   1.2268e-14   1.0000e+00
    
    

### Common functions


```matlab
% Change Directory
cd 'C:\jcabelloc\workspace\jupyter-notebooks\octave\learning_octave';
% List files in the current directory
ls;
% Loading data separed by commas 
data = load('dataxy.txt');
```

     Volume in drive C has no label.
     Volume Serial Number is 3C46-9A6F
    
     Directory of C:\jcabelloc\workspace\jupyter-notebooks\octave\learning_octave
    
    [.]                            dataxy.txt
    [..]                           octave_getting_started.ipynb
    [.ipynb_checkpoints]           
                   2 File(s)         10,922 bytes
                   3 Dir(s)  69,180,248,064 bytes free
    


```matlab
% Ask for help
% help rand
% help randn
```
