"""
Test on importing data from excel files
"""
import os
import numpy as np
from nextorch import io

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, 'test_input.xlsx')

data, data_full = io.read_excel(file_path)
X, Y, X_names, Y_names = io.split_X_y(data_full, Y_names = 'Yield')

def test_X():
    assert np.all(np.array([3.e+02, 1.e-01, 2.e-01, 9.e-01]) == X[0,:])

def test_Y():
    assert np.all(np.array([0.25, 0.665, 0.5, 0.52, 0.5454, 0.5451, 0.9]) == Y[:,0])

def test_X_nams():
    assert np.all(['Temperature', 'Pressure', 'Concentration_1', 'Concentration_2'] == X_names)

def test_Y_names():
    assert np.all(['Yield'] == Y_names)
