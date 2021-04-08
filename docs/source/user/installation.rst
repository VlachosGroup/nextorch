================
Installation
================

Installing Python
-----------------
Anaconda is the recommended method to install Python for scientific
applications. It is supported on Linux, Windows and Mac OS X.
`Download Anaconda here`_. Note that NEXTorch runs on Python 3.7 and above.

Creating a new conda environment (optional)
--------------------------------------------
It may not be possible for one Python installation to meet the requirements of every application. 
To avoid such conflicts, we recommend working inside a virtual environment dedicated to all PyTorch applications. 
If you have not done so, use `conda` to create a new environment:

.. code-block::

    conda create -n torch 

Activate the new environment:

.. code-block::

    conda activate torch

Deactivate it after each use:

.. code-block::

    conda deactivate


Installing NEXTorch using pip
---------------------------------
Using `pip` is the most straightforward way to install NEXTorch.

1. Activate the virtual environment for PyTorch.

2. Install NEXTorch by typing the following in the command prompt (the fresh installation takes ~1-2 minutes):
   
.. code-block::

    pip install nextorch


Installing NEXTorch from source
----------------------------------
If you would prefer to install from source or you are interested in development,
follow the instructions below.

.. code-block::

    pip install git+https://github.com/VlachosGroup/nextorch.git


Upgrading NEXTorch using pip
-------------------------------
To upgrade to a newer release, use the --upgrade flag:

.. code-block::

    pip install --upgrade nextorch


Running unit tests
------------------
NEXTorch has a suite of unit tests built on the `pytest` framework. One should run the tests to ensure all code functions as expected. 
Run the following commands in a Python terminal (usually takes less than a minute):

.. code-block::

     pytest --pyargs nextorch

The expected output is shown below. The number of tests will not
necessarily be the same.

::

    PACKAGE_PATH\nextorch\test\test_1d_function.py ..                                     [ 15%]
    PACKAGE_PATH\nextorch\test\test_EHVI.py ..                                            [ 30%]
    PACKAGE_PATH\nextorch\test\test_io.py ....                                            [ 61%]
    PACKAGE_PATH\nextorch\test\test_parameter.py ...                                      [ 84%]
    PACKAGE_PATH\nextorch\test\test_plotting.py ..                                        [100%]

    ================================== 13 passed in 44.00s ======================================

.. _`Download Anaconda here`: https://www.anaconda.com/distribution/#download-section
