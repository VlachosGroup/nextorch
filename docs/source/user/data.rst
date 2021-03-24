=============================
Data Type and Preprocessing
=============================

.. currentmodule:: nextorch.utils


Data Types 
------------
Python list_, numpy.ndarray_, and torch.tensor_ (PyTorch tensors) are the base data types. 
The :code:`Experiment` class can handle data saved in any base type.

We use several customized types to specify the dimensionality and the base type of the data.

=======================  ===================================================== 
Customized Type           Base Type
=======================  ===================================================== 
:code:`Array`             1D numpy.ndarray_
:code:`Matrix`            2D numpy.ndarray_
:code:`ArrayLike1d`      1D numpy.ndarray_, list_, or torch.tensor_
:code:`MatrixLike2d`     2D numpy.ndarray_, list_, or torch.tensor_
=======================  ===================================================== 

Usually, users provide data in numpy.ndarray_ or list_. NEXTorch converts them to torch.tensor_ and then pass them between BoTorch functions. 
Once the training is done, NEXTorch converts the data to numpy.ndarray_ for output and visualization purposes.

Type Conversion 
-----------------

.. autosummary::
    :nosignatures:

    np_to_tensor
    tensor_to_np
    expand_list


(Inverse) Normalization
--------------------------
Convert arrays or matrics from a real scale into a unit scale [0, 1] in each dimension, vice versa. 
This step is often needed for :code:`X`. 

.. autosummary::
    :nosignatures:

    unitscale_xv
    unitscale_X
    inverse_unitscale_xv
    inverse_unitscale_X



(Inverse) Standardization
--------------------------
Convert arrays or matrics from a real scale into a standardized scale with a zero mean and unit variance in each dimension, vice versa.
This step is often needed for :code:`Y`. 

.. autosummary::
    :nosignatures:

    standardize_X
    inverse_standardize_X


Test Points Generation
----------------------------
Generate :code:`X` points as in a mesh grid for visualization or testing purposes. 

.. autosummary::
    :nosignatures:

    create_X_mesh_2d
    transform_Y_mesh_2d
    transform_X_2d
    prepare_full_X_unit
    prepare_full_X_real
    get_baseline_unit
    fill_full_X_test
    create_full_X_test_2d
    create_full_X_test_1d



Encoding/Decoding
------------------
For ordinal and categorical variables, their real values need to be converted into unit-scale encodings in the continuous space. 
We can do it with :code:`real_to_encode_X`. These encodings are used to train BO functions. 

To convert the unit-scale encodings back to the original variable values, we can do it in two steps: using :code:`unit_to_encode_X` and :code:`encode_to_real_X`.

A :code:`ParameterSpace` class can also be the input to these functions.

.. autosummary::
    :nosignatures:

    encode_xv
    decode_xv
    real_to_encode_X
    unit_to_encode_X
    encode_to_real_X
    real_to_encode_ParameterSpace
    unit_to_encode_ParameterSpace
    encode_to_real_ParameterSpace



.. _list: https://docs.python.org/3/library/stdtypes.html?highlight=list#lists
.. _numpy.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
.. _torch.tensor: https://pytorch.org/docs/stable/tensors.html