==================
Input and Output 
==================

NEXTorch can read data from CSV or Excel files. It can also output the same format. 

IO functions that interfaces with CSV or Excel are built on pandas_. 
The data are first read in as pandas.DataFrame_. They are then converted to numpy.ndarray_, passed to :code:`Experiment` classes and used for plotting. 
Data conversion between numpy.ndarray_ and torch.tensor_ (PyTorch tensors) are handled automatically insided :code:`Experiment` classes.


Examples
---------

Specify the X and Y variable names in :code:`X_names` and :code:`Y_names` as Python lists. 
Read the data from a CSV file.

.. code-block:: python

    from nextorch import io

    var_names = X_names + Y_names
    data, data_full = read_csv(file_path, var_names = var_names)
    X_real, Y_real, _, _ = io.split_X_y(data, Y_names = Y_names)



Convert the output data from a numpy.ndarray_ to a pandas.DataFrame_ and then export it to excel.

.. code-block:: python

    from nextorch import io

    data = io.np_to_dataframe([X, Y], var_names)
    data.to_csv('test_data.csv')


.. _pandas: https://pandas.pydata.org/
.. _pandas.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _numpy.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
.. _torch.tensor: https://pytorch.org/docs/stable/tensors.html