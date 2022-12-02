#!/usr/bin/env python
# coding: utf-8

# In[54]:


# The first few lines of code is from online https://stackoverflow.com/questions/36816537/spherical-coordinates-plot-in-matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

theta, phi = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)
R = 1 + 0.1*np.sin(7*PHI)*np.cos(THETA)
X = R * np.sin(PHI) * np.cos(THETA)
Y = R * np.sin(PHI) * np.sin(THETA)
Z = R * np.cos(PHI)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
    linewidth=0, antialiased=False, alpha=0.5)

plt.show()

R = np.reshape(R, (1, 10000))
X_vec = np.reshape(X, (1, 10000))
Y_vec = np.reshape(Y, (1, 10000))
Z_vec = np.reshape(Z, (1, 10000))
#A matrix of all x values
conc = np.array([X_vec, Y_vec, Z_vec])
conc = np.squeeze(conc, 1)

#apply rotation
#rotation_z = np.matmul([[math.cos(2), -math.sin(2), 0], [math.sin(2), math.cos(2), 0], [0, 0, 1]], [[math.cos(2), 0, -math.sin(2)], [0, 1, 0], [math.cos(2), 0, math.sin(2)]])
rotation_z = [[math.cos(2), -math.sin(2), 0], [math.sin(2), math.cos(2), 0], [0, 0, 1]]
conc_r = np.matmul(rotation_z, conc)
X_vec_1 = conc_r[0,:]
Y_vec_1 = conc_r[1,:]
Z_vec_1 = conc_r[2,:]
X = np.reshape(X_vec_1, (100, 100))
Y = np.reshape(Y_vec_1, (100, 100))
Z = np.reshape(Z_vec_1, (100, 100))
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
    linewidth=0, antialiased=False, alpha=0.5)

plt.show()

def g(x):
    lsdiff = 1/10000*np.sum(np.square(np.matmul([[math.cos(x[0])*math.cos(x[1]), math.cos(x[0])*math.sin(x[1])*math.sin(x[2]) - math.sin(x[0])*math.cos(x[2]), math.cos(x[0])*math.sin(x[1])*math.cos(x[2]) + math.sin(x[0])*math.sin(x[2])],
                 [math.sin(x[0])*math.cos(x[1]), math.sin(x[0])*math.sin(x[1])*math.sin(x[2]) + math.cos(x[0])*math.cos(x[2]), math.sin(x[0])*math.sin(x[1])*math.cos(x[2]) - math.cos(x[0])*math.sin(x[2])],
                 [-math.sin(x[1]), math.cos(x[1])*math.sin(x[2]), math.cos(x[1])*math.cos(x[2])]], conc_r) - conc))
    return lsdiff
def gradg(x):
    dmdx_0my = np.matmul([[-math.sin(x[0])*math.cos(x[1]), -math.sin(x[0])*math.sin(x[1])*math.sin(x[2]) - math.cos(x[0])*math.cos(x[2]), -math.sin(x[0])*math.sin(x[1])*math.cos(x[2]) + math.cos(x[0])*math.sin(x[2])],
              [math.cos(x[0])*math.cos(x[1]),  math.cos(x[0])*math.sin(x[1])*math.sin(x[2]) - math.sin(x[0])*math.cos(x[2]),  math.cos(x[0])*math.sin(x[1])*math.cos(x[2]) + math.sin(x[0])*math.sin(x[2])],
              [0,                                                                        0,                                                                       0]], conc_r)
    dmdx_1my = np.matmul([[-math.cos(x[0])*math.sin(x[1]), math.cos(x[0])*math.cos(x[1])*math.sin(x[2]), math.cos(x[0])*math.cos(x[1])*math.cos(x[2])],
               [-math.sin(x[0])*math.sin(x[1]), math.sin(x[0])*math.cos(x[1])*math.sin(x[2]), math.sin(x[0])*math.cos(x[1])*math.cos(x[2])],
               [-math.cos(x[1]),               -math.sin(x[1])*math.sin(x[2]),               -math.sin(x[1]) * math.cos(x[2])]], conc_r)
    dmdx_2my = np.matmul([[0, math.cos(x[0])*math.sin(x[1])*math.cos(x[2]) + math.sin(x[0])*math.sin(x[2]), -math.cos(x[0])*math.sin(x[1])*math.sin(x[2]) + math.sin(x[0])*math.cos(x[2])],
              [0, math.sin(x[0])*math.sin(x[1])*math.cos(x[2]) - math.cos(x[0])*math.sin(x[2]), -math.sin(x[0])*math.sin(x[1])*math.sin(x[2]) - math.cos(x[0])*math.cos(x[2])],
              [0, math.cos(x[1])*math.cos(x[2]),                                               -math.cos(x[1])*math.sin(x[2])]], conc_r)
    lsdiffd = 2/10000*(np.matmul([[math.cos(x[0])*math.cos(x[1]), math.cos(x[0])*math.sin(x[1])*math.sin(x[2]) - math.sin(x[0])*math.cos(x[2]), math.cos(x[0])*math.sin(x[1])*math.cos(x[2]) + math.sin(x[0])*math.sin(x[2])],
                 [math.sin(x[0])*math.cos(x[1]), math.sin(x[0])*math.sin(x[1])*math.sin(x[2]) + math.cos(x[0])*math.cos(x[2]), math.sin(x[0])*math.sin(x[1])*math.cos(x[2]) - math.cos(x[0])*math.sin(x[2])],
                 [-math.sin(x[1])              , math.cos(x[1])*math.sin(x[2])                                               , math.cos(x[1])*math.cos(x[2])]], conc_r) - conc)
    gradient = [np.sum(np.multiply(lsdiffd, dmdx_0my)), np.sum(np.multiply(lsdiffd, dmdx_1my)), np.sum(np.multiply(lsdiffd, dmdx_2my))]
    return gradient

# In[ ]:
#Pick a random point from a unit sphere
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec = vec / np.linalg.norm(vec, axis=0)
    return vec
def sample_angle(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec = vec % (2*math.pi)
    return vec
#g is the function I am going to minimize
def steepest_descent(x, tol, N):
    k = 1
    g_1 = np.zeros(len(x))
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
                #print("no likely improvement", k)
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
        #no normalization for normal gradient descent
        #x = x/np.linalg.norm(x)
        #book algorithm says tol instead of tol/100
        if abs(g(x) - g_1) < tol:
            #print("success", k)
            return x, g(x)
        k = k + 1
    print ("maximum iterations exceeded")
    return x, g_1

#set a small tolerance
tol = 0.00005
#100000 samples
num_of_samples = 10000
#1000000 iterations
N = 10000
#sample num_of_samples points
x_table = sample_angle(num_of_samples)
#run steepest descent of the intial value
x = x_table[:, 0]
print("initlal g", g(x))
x, g_x = steepest_descent(x, tol, N)
print("initial guess after descent", x, g_x)
x_min = x
g_min = g_x
for i in range(1, num_of_samples):
    x = x_table[:, i]
    x, g_x = steepest_descent(x, tol, N)
    if g_x < g_min:
        g_min = g_x
        x_min = x
print("min=", x_min, g_min)






