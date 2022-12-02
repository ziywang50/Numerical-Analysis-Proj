# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import math
#Pick a random point from a unit sphere
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec = vec / np.linalg.norm(vec, axis=0)
    return vec
#g is the function I am going to minimize
def g(x):
    sum_value = 0
    for i in range(0, len(x)):
        sum_value = sum_value + x[i]**2 + math.exp(x[i])
    return sum_value
#gradient of g
def gradg(x):
    sum_value = 0
    for i in range(0, len(x)):
        sum_value = sum_value + 2*x[i] + math.exp(x[i])
    return sum_value
def steepest_descent(x, tol, N):
    k = 1
    g_1 = np.zeros(len(x))
    y = np.zeros(len(x))
    while k <= N:
        # compute the value of alpha
        g_1 = g(x)
        z = gradg(x)
        z_0 = np.linalg.norm(gradg(x))
        if z_0 == 0:
            print("zero gradient")
            y = x/np.linalg.norm(x)
            return y, g(y)
        z = z/z_0
        alpha_1 = 0
        alpha_3 = 1
        g_3 = g(x-alpha_3*z)
        while g_3 >= g_1:
            alpha_3 = alpha_3 / 2
            g_3 = g(x-alpha_3*z)
            if alpha_3 < tol/2:
                #print("no likely improvement")
                x = x / np.linalg.norm(x)
                return x, g(x)
        alpha_2 = alpha_3/2
        g_2 = g(x-alpha_2*z)
        #quadratic interpolation
        h_1 = (g_2 - g_1)/alpha_2
        h_2 = (g_3 - g_2)/(alpha_3 - alpha_2)
        h_3 = (h_2 - h_1)/alpha_3
        alpha_0 = 0.5*(alpha_2 - h_1/h_3)
        g_0 = g(x-alpha_0*z)
        #finding the best value of alpha
        best_alpha = alpha_0
        if g(x-alpha_3*z) < g_0:
            best_alpha = alpha_3
        x = x - best_alpha*z
        x = x/np.linalg.norm(x)
        if abs(g(x) - g_1) < tol:
            print("success")
            return x, g(x)
        k = k + 1
    print ("maximum iterations exceeded")
    return x, g_1

#set a small tolerance
tol = 0.005
num_of_samples = 100000
#100000 iterations
N = 100000
#sample num_of_samples points
x_table = sample_spherical(num_of_samples)
#run steepest descent of the intial value
x = x_table[:, 0]
x, g_x = steepest_descent(x, tol, N)
print("initial guess", x, g_x)
x_min = x
g_min = g_x
for i in range(1, N):
    x = x_table[:, i]
    x, g_x = steepest_descent(x, tol, N)
    if g_x < g_min:
        g_min = g_x
        x_min = x
print(x_min, g_min)

#matplotlib 3d plot
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
