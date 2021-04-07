# import os
# import sys
# project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_path)

from nextorch import io


data, data_full = io.read_excel('test_input.xlsx')
X,  Y, X_names, Y_names = io.split_X_y(data_full, Y_names = 'Yield')
