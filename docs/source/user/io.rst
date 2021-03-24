==================
Input and Output 
==================

.. currentmodule:: nextorch.io

NEXTorch can read data from CSV or Excel files. It can output the same format. 

IO functions that interfaces with CSV or Excel are built on pandas_. 
The data are first read in as pandas.DataFrame_. They are then converted to numpy.ndarray_, passed to :code:`Experiment` classes and used for plotting. 

Examples
---------

Specify the X and Y variable names in :code:`X_names` and :code:`Y_names` as Python lists. 
Read the data from a CSV file.

.. code-block:: python

    from nextorch import io

    var_names = X_names + Y_names
    data, data_full = read_csv(file_path, var_names = var_names)
    X_real, Y_real, _, _ = io.split_X_y(data, Y_names = Y_names)



Convert the output data from a numpy.ndarray_ to a pandas.DataFrame_ and then export it to a CSV file.

.. code-block:: python

    from nextorch import io

    data = io.np_to_dataframe([X, Y], var_names)
    data.to_csv('test_data.csv')



Here is a list of IO functions in :code:`nextorch.io` module.

.. autosummary::
    :nosignatures:

    np_to_dataframe
    read_csv
    read_excel
    split_X_y

.. _pandas: https://pandas.pydata.org/
.. _pandas.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _numpy.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html



