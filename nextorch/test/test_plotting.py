
"""
Test 3d plotting and creating test matrices
"""
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from nextorch import plotting, utils

dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path, 'test_3d.png')

# test adding a 2d plane to a 3d space
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
plotting.add_x_slice_2d(ax, 10, [1,100], [2,4])
fig.savefig(save_path, bbox_inches="tight")

# switch back to interactive mode
# matplotlib.use('TkAgg')

# test create full X functions
X_ranges = [[0,1], [0,2], [0,3], [2,3]]
X_test_2d, _, _ = utils.create_full_X_test_2d(X_ranges=X_ranges, x_indices=[0,1]) 
X_test_1d = utils.create_full_X_test_1d(X_ranges=X_ranges, x_index=0, baseline = 'center') 


def test_X_2d_dimensions():
    # Test on two-dimensional testing matrix
    assert X_test_2d.shape[0] == 41**2
    assert X_test_2d.shape[1] == len(X_ranges)


def test_X_1d_values():
    # Test on one-dimensional testing matrix
    x_real_0 = np.linspace(0,1, 41)
    assert np.all(x_real_0 == X_test_1d[:,0])
    assert np.all(X_test_1d[:,1:] == 0.5)
