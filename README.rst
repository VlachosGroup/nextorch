=========
NEXTorch
=========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5144404.svg
   :target: https://doi.org/10.5281/zenodo.5144404

NEXTorch is an open-source software package in Python/PyTorch to faciliate 
experimental design using Bayesian Optimization (BO). 

NEXTorch stands for Next EXperiment toolkit in PyTorch/BoTorch. 
It is also a library for learning the theory and implementation of Bayesian Optimization.


.. image:: https://github.com/VlachosGroup/nextorch/blob/62b6163d65d2b49fdb8f6d3485af3222f4409500/docs/source/logos/nextorch_logo_doc.png

Documentation
-------------

See our `documentation page`_ for examples, equations used, and docstrings.


Developers
----------

-  Yifan Wang (wangyf@udel.edu)
-  Tai-Ying (Chris) Chen

Dependencies
------------

-  Python >= 3.7
-  `PyTorch`_ >= 1.8: Used for tensor operations with GPU and autograd support
-  `GPyTorch`_ >= 1.4: Used for training Gaussian Processes
-  `BoTorch`_ = 0.4.0: Used for providing Bayesian Optimization framework
-  `Matplotlib`_: Used for generating plots
-  `PyDOE2`_: Used for constructing experimental designs
-  `Numpy`_: Used for vector and matrix operations
-  `Scipy`_: Used for curve fitting
-  `Pandas`_: Used to import data from Excel or CSV files
-  `openpyxl`_: Used by Pandas to import Excel files
-  `pytest`_: Used for unit tests


.. _documentation page: https://nextorch.readthedocs.io/en/latest/
.. _PyTorch: https://pytorch.org/
.. _GPyTorch: https://gpytorch.ai/ 
.. _BoTorch: https://botorch.org/
.. _Matplotlib: https://matplotlib.org/
.. _pyDOE2: https://pythonhosted.org/pyDOE/
.. _Numpy: http://www.numpy.org/
.. _Scipy: https://www.scipy.org/
.. _Pandas: https://pandas.pydata.org/
.. _openpyxl: https://openpyxl.readthedocs.io/en/stable/
.. _pytest: https://docs.pytest.org/en/stable/



Getting Started
---------------

1. Install using pip (see documentation for full instructions)::

    pip install nextorch

2. Run the unit tests.

3. Read the documentation for tutorials and examples.


License
-------

This project is licensed under the MIT License - see the LICENSE.md.
file for details.


Contributing
------------

If you have a suggestion or find a bug, please post to our `Issues` page on GitHub. 

Questions
---------

If you are having issues, please post to our `Issues` page on GitHub.

Funding
-------

This material is based upon work supported by the Department of Energy's Office 
of Energy Efficient and Renewable Energy's Advanced Manufacturing Office under 
Award Number DE-EE0007888-9.5.

Acknowledgements
------------------

-  Jaynell Keely (Logo design)
  

Publications
------------

\Y. Wang, T.-Y. Chen, and D.G. Vlachos, NEXTorch: A Design and Bayesian Optimization Toolkit for Chemical Sciences and Engineering, J. Chem. Inf. Model. 2021, 61, 11, 5312–5319.
