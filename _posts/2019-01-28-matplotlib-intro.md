---
layout: post
title:  "Introduction to Matplotlib"
date:   2019-01-27 21:20:34 -0500
categories: [Python]
tags: [python, matplotlib, introduction]
---


A brief introduction to Matplotlib

"Matplotlib is a Python 2D plotting library which produces publication quality figures."


```python
# Import pyplot and numpy
import matplotlib.pyplot as plt
import numpy as np
```


```python
# A simple plot y = x
x = np.linspace(0, 2, 100)
plt.plot(x, x)
```




    [<matplotlib.lines.Line2D at 0x3a5e6a8390>]




![png](/assets/2019-01-28-matplotlib-intro/output_2_1.png)



```python
# A simple plot with multiple lines
x = np.linspace(0, 2, 100)
plt.plot(x, x, label='linear')
plt.plot(x, x**0.5, label='square_root')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()
```


![png](/assets/2019-01-28-matplotlib-intro/output_3_0.png)


### Intro to pyplot


```python
# Basic plot
import matplotlib.pyplot as plt
plt.plot([10, 30, 20, 40, 30, 50])
plt.ylabel('some numbers')
plt.show()
```


![png](/assets/2019-01-28-matplotlib-intro/output_5_0.png)



```python
# Basic plot with x values
plt.plot([11, 12, 13, 14, 15, 16], [10, 30, 20, 40, 30, 50])
plt.show()
```


![png](/assets/2019-01-28-matplotlib-intro/output_6_0.png)



```python
# Format the style of a plot
plt.plot([11, 12, 13, 14, 15, 16], [10, 30, 20, 40, 30, 50], 'bx')
plt.axis([10, 20, 0, 100]) # [xmin, xmax, ymin, ymax]
plt.show()
```


![png](/assets/2019-01-28-matplotlib-intro/output_7_0.png)



```python
# Plot many datasets
t = np.arange(0., 5., 0.2)
# red line, yellow squares, green triangles and blue dashes
plt.plot(t, t, 'r-', t, t**2, 'ys', t, t**3, 'g^', t, t**0.5, 'b--')
plt.show()
```


![png](/assets/2019-01-28-matplotlib-intro/output_8_0.png)



```python
# Control line properties
x = np.linspace(1,10, 100)
plt.plot(x, linewidth=4.0)
```




    [<matplotlib.lines.Line2D at 0x3a5e24f908>]




![png](/assets/2019-01-28-matplotlib-intro/output_9_1.png)



```python
# Plot with categorical variables
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(1, figsize=(9, 3)) # wigth and height in inches

plt.subplot(1, 3, 1)  # nrows, ncols, plot_number
plt.bar(names, values)
plt.subplot(1, 3, 2)
plt.scatter(names, values)
plt.subplot(1, 3, 3)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()
```


![png](/assets/2019-01-28-matplotlib-intro/output_10_0.png)



```python
# Work with multiple subplots
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(2, 1, 2)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
```


![png](/assets/2019-01-28-matplotlib-intro/output_11_0.png)



```python
# create multiple figures by using multiple figure() calls
plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])


plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3') # subplot 211 title
```




    Text(0.5, 1.0, 'Easy as 1, 2, 3')




![png](/assets/2019-01-28-matplotlib-intro/output_12_1.png)



![png](/assets/2019-01-28-matplotlib-intro/output_12_2.png)



```python
# Add Text in an arbritrary location
x = np.linspace(0,10,100)
y = 2 + 2 * x
plt.text(5, 10, r'$y = 2 + 2x $')
plt.text(5, 15, r"$\theta_0 = 2$", fontsize=14, color="r")
plt.text(5, 20, r"$\theta_1 = 0.2 \times 10^{1}$", fontsize=14, color="r")
plt.axis([0, 10, 0, 30])
plt.plot(x, y)

```




    [<matplotlib.lines.Line2D at 0x425e7be9b0>]




![png](/assets/2019-01-28-matplotlib-intro/output_13_1.png)



```python
# Annotate text
ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
plt.plot(t, s, linewidth=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.plot([2, 2], [-1.5, 1], "r--")

plt.axis([0, 6, -1.5, 2])
plt.show()
```


![png](/assets/2019-01-28-matplotlib-intro/output_14_0.png)


### Matplotlib object-oriented interface


```python
# Generate datasets
x = np.linspace(0, 4, 100)
y = np.sin(2*np.pi*x)
# Create Figure
fig = plt.figure()

# Add axes to the figure
# The dimensions [left, bottom, width, height] of the new axes. 
# All quantities are in fractions of figure width and height.(range 0 to 1)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 

# Plot on that set of axes
axes.plot(x, y, 'b--')
axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods
axes.set_ylabel('Set y Label')
axes.set_title('sin() function')
```




    Text(0.5, 1.0, 'sin() function')




![png](/assets/2019-01-28-matplotlib-intro/output_16_1.png)



```python
# Generate datasets
x = np.linspace(0, 4, 100)
y = np.sin(2*np.pi*x)

# Creates blank canvas
fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# Larger Figure Axes 1
axes1.plot(x, x**x, 'k')
axes1.set_xlabel('X_2')
axes1.set_ylabel('x^2')
axes1.set_title('Axes 1 Title')

# Insert Figure Axes 2
axes2.plot(x, y, 'g--')
axes2.set_xlabel('X_2')
axes2.set_ylabel('sin(x)')
axes2.set_title('Axes 2 Title');
```


![png](/assets/2019-01-28-matplotlib-intro/output_17_0.png)



```python
# Use subplots
fig, axes = plt.subplots()

# Use the axes object to add features
axes.plot(x, y, 'b')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('sin(x)');
```


![png](/assets/2019-01-28-matplotlib-intro/output_18_0.png)



```python
# Use subplots
fig, axes = plt.subplots(nrows=1, ncols=2)
x = np.linspace(0, 4, 100)

# tight_layout automatically adjusts subplot params 
# so that the subplot(s) fits in to the figure area
fig.tight_layout()

axes[0].plot(x, np.sin(2*np.pi*x), 'b')
axes[1].plot(x, np.cos(2*np.pi*x), 'r')
axes[0].set_title('sin(x)')
axes[1].set_title('cos(x)')
```




    Text(0.5, 1.0, 'cos(x)')




![png](/assets/2019-01-28-matplotlib-intro/output_19_1.png)



```python
# Set figure size
fig, axes = plt.subplots(figsize=(8,2))

axes.plot(x, y, 'g*')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('sin(x)');
```


![png](/assets/2019-01-28-matplotlib-intro/output_20_0.png)



```python
# Save figures
fig.savefig("sin_function.png")
```


```python
# Show legends
x = np.linspace(0, 2, 100)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.plot(x, x**2, label="Square x^2")
ax.plot(x, x**3, label="Cubic x^3")
ax.legend()
```




    <matplotlib.legend.Legend at 0x3a634f96a0>




![png](/assets/2019-01-28-matplotlib-intro/output_22_1.png)



```python
# Set the range of the axes
x = np.linspace(0, 6, 100)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x, x**2, x, x**3)
axes[0].set_title("default axes ranges")

axes[1].plot(x, x**2, x, x**3)
axes[1].axis('tight') # Auto tight
axes[1].set_title("tight axes")

axes[2].plot(x, x**2, x, x**3)
axes[2].set_ylim([0, 60])
axes[2].set_xlim([1, 5])
axes[2].set_title("custom axes range");
```


![png](/assets/2019-01-28-matplotlib-intro/output_23_0.png)


### Plot from a Dataframe


```python
import pandas as pd
df = pd.DataFrame(np.random.randn(10, 4), index=list('abcdefghij'),
             columns=list('ABCD'))
df.plot(kind='scatter', x="A", y='B')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb91b814dd8>




![png](/assets/2019-01-28-matplotlib-intro/output_25_1.png)



```python
# Plot the histogram for each numerical value
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.show()
```


![png](/assets/2019-01-28-matplotlib-intro/output_26_0.png)


### Samples of plots

### References:
https://matplotlib.org/


