================
Installation
================

Installing Python
-----------------
Anaconda is the recommended method to install Python for scientific
applications. It is supported on Linux, Windows and Mac OS X.
`Download Anaconda here`_. Note that NEXTorch runs on Python 3.X.

Installing PyTorch, BoTorch and GPyTorch
------------------------------------------
Using `conda` is the recommended method.
.. code-block::

    conda install botorch -c pytorch -c gpytorch


Installing NEXTorch using pip
---------------------------------
Using `pip` is the most straightforward way to install NEXTorch.

1. Open a command prompt with access to Python (if Python is installed via
   Anaconda on Windows, open the Anaconda Prompt from the start menu).

2. Install NEXTorch by typing the following in the command prompt:
.. code-block::

    pip install nextorch

The output towards the end should state "Successfully built nextorch" if the
installation was successful. 


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
NEXTorch has a suite of unit tests that should be run before committing any code.
To run the tests, run the following commands in a Python terminal.
.. code-block::

     import nextorch
     nextorch.run_tests()

The expected output is shown below. The number of tests will not
necessarily be the same. ::

    .........................
    ----------------------------------------------------------------------
    Ran 25 tests in 0.020s

    OK

.. _`Download Anaconda here`: https://www.anaconda.com/distribution/#download-section
