# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:31:19 2019

@author: wangyf
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

colormap = cm.summer
def plot_2D_x_slice(ax, xvalue, yrange, zrange, mesh_size = 100):

    Y, Z = np.meshgrid(np.linspace(yrange[0], yrange[1], mesh_size), np.linspace(zrange[0], zrange[1], mesh_size), indexing = 'ij')
    X = xvalue * np.ones((mesh_size, mesh_size))
    ax.plot_surface(X, Y, Z,  cmap=colormap, rstride=1 , cstride=1, shade=False, alpha = 0.7)

 
    
def plot_2D_y_slice(ax, xrange, yvalue, zrange, mesh_size = 100):

    X, Z = np.meshgrid(np.linspace(xrange[0], xrange[1], mesh_size), np.linspace(zrange[0], zrange[1], mesh_size), indexing = 'ij')
    Y = yvalue * np.ones((mesh_size, mesh_size))
    ax.plot_surface(X, Y, Z,  cmap=colormap, rstride=1 , cstride=1, shade=False, alpha = 0.7)

    

def plot_2D_z_slice(ax, xrange, yrange, zvalue, mesh_size = 100):

    X,Y = np.meshgrid(np.linspace(xrange[0], xrange[1], mesh_size), np.linspace(yrange[0], yrange[1], mesh_size), indexing = 'ij')
    Z = zvalue * np.ones((mesh_size, mesh_size))
    ax.plot_surface(X, Y, Z,  cmap=colormap, rstride=1 , cstride=1, shade=False, alpha = 0.7)

# Examples
#plot_2D_x_slice(10, [1,100], [2,4])
#plot_2D_y_slice([10,30], 100, [2,4])
#plot_2D_z_slice([10,30], [1,100], 0.8)
       