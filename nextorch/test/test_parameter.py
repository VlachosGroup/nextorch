"""
Test parameter and parameterspace
"""
import numpy as np

from nextorch import io, bo, doe
import nextorch.utils as ut
from nextorch.parameter import Parameter, ParameterSpace


# p1 - ordinal
p1 = Parameter(x_type = 'ordinal', interval = 1, x_range = [0,2])
p1_encoding_unit = ut.unitscale_xv(p1.encoding, p1.x_range)

def test_param_ordinal():
    assert np.all(p1_encoding_unit == [0, 0.5, 1])

# p2 - categorical
p2_values = ['low', 'medium low', 'medium', 'medium high', 'high']
p2 = Parameter(x_type = 'categorical', values = p2_values)
p2_encoding_unit = ut.unitscale_xv(p2.encoding, p2.x_range)

def test_param_categorical():
    assert np.all(p2_encoding_unit == [0, 0.25, 0.5, 0.75, 1])

#%% Use ParameterSpace
Xunit = doe.full_factorial([5,5])
xi = Xunit[:,0]
Y = np.array([1, 2, 3])

ps = ParameterSpace([p1, p2])
X_encode = ut.unit_to_encode_ParameterSpace(Xunit, ps)
X_real = ut.encode_to_real_ParameterSpace(X_encode, ps)

#%% Use database copy
ds = bo.Database()
ds.define_space([p1, p2])
ds.input_data(Xunit, Y, X_names = ['lol', 'ox'], unit_flag=True)

def test_database():
    X_true = np.array([[0.0, 'low'], [0.0, 'low'], [1.0, 'low'], [1.0, 'low'], [2.0, 'low'], 
    [0.0, 'medium low'], [0.0, 'medium low'], [1.0, 'medium low'], [1.0, 'medium low'], [2.0, 'medium low'], 
    [0.0, 'medium'], [0.0, 'medium'], [1.0, 'medium'], [1.0, 'medium'], [2.0, 'medium'], 
    [0.0, 'medium high'], [0.0, 'medium high'], [1.0, 'medium high'], [1.0, 'medium high'], [2.0, 'medium high'],
    [0.0, 'high'], [0.0, 'high'], [1.0, 'high'], [1.0, 'high'], [2.0, 'high']], dtype=object)

    assert np.all(X_real == X_true)
    assert np.all(ds.X_real == X_true)