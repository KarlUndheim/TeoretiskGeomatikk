# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:30:55 2022

@author: Karl
"""
import numpy as np
from numpy.linalg import multi_dot, inv
import math

# task a

# Observations
n = 21

# Unknowns
e = 6


# Data from inital table is to be modified with regard to new origin and stored here.
# Columns are x, y, N and P
data = np.zeros((21,4))

# Array with ellipsoidal heights, which is to be used in later tasks, but is filled now.
ellipsoidal_heights = np.zeros(n)

# We pick point 7 as origin.
# The file "points.txt" is a copy of the table given in the assignment text.
with open('points.txt') as file:
    for i, line in enumerate(file):
        if i==6:
            x_origin,y_origin = [float(s) for s in line.split()[1:3]]
            break
    file.close()
    
# Fill the data matrix, and ellipsoidal heights
with open('points.txt') as file:
    for i, line in enumerate(file):
        x,y,h_ell,h_nn = [float(s) for s in line.split()[1:-1]]
        method = line.split()[-1]
        data[i][0] = x - x_origin
        data[i][1] = y - y_origin
        data[i][2] = h_ell - h_nn
        data[i][3] = (1.0 if method=="Trig" else 4.0)
        ellipsoidal_heights[i] = h_ell
        
# Matrices to be used in calculations
A = np.zeros((n, e))
l = np.zeros(n)
P = np.zeros((n, n))

# Fill these matrices
for i in range(len(data)):
    x = data[i][0]
    y = data[i][1]
    A[i][0] = x**2
    A[i][1] = x*y
    A[i][2] = y**2
    A[i][3] = x
    A[i][4] = y
    A[i][5] = 1.0
    l[i] = data[i][2]
    P[i][i] = data[i][3]
    
N = multi_dot([A.T, P, A])

# These are the estimated parameters. 
x = multi_dot([inv(N), A.T, P, l])

names = "ABCDEF"

print("Estimated paramaters:")
for i, element in enumerate(x):
    print(names[i]+": "+str(element))

v = np.dot(A, x) - l
sigma0squared = multi_dot([v.T, P, v])/(n-e)
covariance = np.dot(sigma0squared, inv(N))

print("")
print("Covariances: ")
print('\n'.join([' '.join([str(round(cell,15)) for cell in row]) for row in covariance]))
print("")

print("Test")
for i in range(6):
    t = abs(x[i]/(math.sqrt(covariance[i][i])))
    print("%.3f" % t)
print("")
    
# task b

# C is removed
e = 5

# We need a new A matrix for the re-restimation
A = np.zeros((n, e))

# Fill A
for i in range(len(data)):
    x = data[i][0]
    y = data[i][1]
    A[i][0] = x**2
    A[i][1] = x*y
    A[i][2] = x
    A[i][3] = y
    A[i][4] = 1.0
    
N = multi_dot([A.T, P, A])

# These are the new estimated parameters. 
x = multi_dot([inv(N), A.T, P, l])

names = "ABDEF"

print("New estimations:")
for i, element in enumerate(x):
    print(names[i]+": "+str(element))

v = np.dot(A, x) - l
sigma0squared = multi_dot([v.T, P, v])/(n-e)
covariance = np.dot(sigma0squared, inv(N))

print("")
print("New covariances: ")
print('\n'.join([' '.join([str(round(cell,15)) for cell in row]) for row in covariance]))
print("")

print("New standard deviations")
for i in range(5):
    print(math.sqrt(covariance[i][i]))
print("")

print("New test")
for i in range(5):
    t = abs(x[i]/(math.sqrt(covariance[i][i])))
    print("%.3f" % t)
    
# task c

deflection_north = -2*x[0]*data[0][0]-x[1]*data[0][1]-x[2]
deflection_east = -x[1]*data[0][1]-x[3]

print("")
print("Deflections of the vertical:")
print(deflection_north)
print(deflection_east)
print("")

# task d

# Array to store the geoid heights
geoid_heights = np.zeros(21)

# Here our model is used to calculate the geoid heights
for i in range(len(data)):
    N = x[0]*data[i][0]**2 + x[1]*data[i][0]*data[i][1] + x[2]*data[i][0] + x[3]*data[i][1] + x[4]
    geoid_heights[i] = N

print("Geoid heights:")
for element in geoid_heights:
    print("%.5f" % element)
print("")
    
print("Residuals:")
v = np.dot(A, x) - l
for element in v:
    print("%.7f" % element)
print("")
    
# Standard deviation of unit weight:
sigma0 = math.sqrt(multi_dot([v.T, P, v])/(n-e))
print("Standard deviation of the unit weight")
print(sigma0)
print("")


# Task e
# Array to store 2008.bin heights
heights_2008 = np.zeros(n)
print("HREF2008a.bin heights:")
with open('2008.txt') as file:
    for i, line in enumerate(file):
        heights_2008[i] = ellipsoidal_heights[i] - float(line)
        print("%.5f" % heights_2008[i])
    file.close()
        
# mean = np.sum(heights_2008)/21

# vv = 0
# for i in range(21):
#     vv += (heights_2008[i] - mean)**2
# deviation = math.sqrt((vv)/(n-1))
# print(deviation)

# task f
# Differences between our local model and the HREF2008a.bin heights
differences = np.zeros(21)
print("")
print("Differences:")
for i in range(21):
    differences[i] = geoid_heights[i] - heights_2008[i]
    print(differences[i])
    
# task h
# Calculation using the equation given in our text.
mean = np.sum(differences)/21

vv = 0
for i in range(21):
    vv += (differences[i]-mean)**2 
deviation = math.sqrt((vv)/(n-1))

print("")
print("Standard deviation of the differences:")
print(deviation)