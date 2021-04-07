from nextorch import io
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, 'test_input.xlsx')

data, data_full = io.read_excel(file_path)
X,  Y, X_names, Y_names = io.split_X_y(data_full, Y_names = 'Yield')
