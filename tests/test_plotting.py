import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from nextorch import plotting, utils
import matplotlib.pyplot as plt

# test add slice functions
fig = plt.figure(figsize=(6, 6))
ax = fig.gca(projection='3d')
plotting.add_x_slice_2d(ax, 10, [1,100], [2,4])


# test create full X functions
X_ranges = [[0,1], [0,2], [0,3], [2,3]]
X_test, _, _ = utils.create_full_X_test_2d(X_ranges=X_ranges, x_indices=[0,1]) 

X_test = utils.create_full_X_test_1d(X_ranges=X_ranges, x_indices=0, baseline = 'center') 