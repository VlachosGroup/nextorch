import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from nextorch import plotting
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6))
ax = fig.gca(projection='3d')
plotting.add_2D_x_slice(ax, 10, [1,100], [2,4])
