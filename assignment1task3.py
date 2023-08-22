# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 00:37:05 2022

@author: Karl
"""

import numpy as np
from numpy.linalg import multi_dot, inv

slopes = np.array([2292.3266, 2033.1163, 1503.6636, 1434.4788, 2705.6997, 
                   2310.1655, 1186.7123, 1888.9902, 1451.5997, 852.1489, 727.1407])

apriori = np.zeros(shape=(11,11))
weight = np.zeros(shape=(11,11))

for i in range(len(slopes)):
    apriori[i] = (5.0 + slopes[i]/1000)
    weight[i][i] = 1/(slopes[i]/1000)

print("Apriori:")
for i in range(len(slopes)):
    print(apriori[i][i])
print("")

print("Weight:")
for i in range(len(slopes)):
    print(weight[i][i])
print("")

A = np.array([[0,0], [1,0], [0,0], [1,0], [0,0], [0,0], [0,1], [0,1], [0,1], [1,0], [1,-1]])

B = multi_dot([inv(multi_dot([A.T, weight, A])), A.T, weight])

Cx = multi_dot([B, apriori, B.T])

S_a = np.sqrt(Cx[0,0])
S_b = np.sqrt(Cx[1,1])

print("Standard deviations:")
print(S_a)
print(S_b)
print("")
print("Covariance matrix:")
print(Cx)