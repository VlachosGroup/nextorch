import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from nextorch import io


data = io.read_excel('test_input.xlsx')
X, X_names, y, y_names = io.split_X_y(data, y_names = ['Concentration_1', 'Yield'] )
