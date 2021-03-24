===============
Overview
===============

Active Learning Framework
--------------------------

Our framework integrates design of experiments (DOE), BO, and surrogate modeling. The goal is to optimize the function of interest, i.e., objective function :math:`f({\bf X})`. 
The objective function can be complex computer simulations or real-world experiments. 

.. image:: ../_images/user/active_learning_framework.svg


:math:`{\bf X}` denotes the input variables (parameters). Here, :math:`{\bf X= x_{1},x_{2},x_{3},…x_{d}}`; 
each of :math:`{\bf x_{i}}` is a vector :math:`{\bf x_{i}} = (x_{1i},x_{2i},…,x_{ni} )^T`
:math:`n` is the number of sample points, and :math:`d` is the dimensionality of the input. 
Each parameter is bounded, and the resulting multidimensional space is called a design space :math:`A`, i.e., :math:`{\bf X} \in A \in \mathbb{R}^{d}`. 

The outputs of f, :math:`{\bf Y}` (responses), are usually expensive, time-consuming, or otherwise difficult to measure.

Initially, a set of initial sampling :code:`X_init` is generated from a DOE. These sampling points are passed to f for evaluation. 

One collects the data :math:`{\bf D= (X,Y)}` and use it to train a cheap-to-evaluate surrogate model :math:`\hat{f}`(a Gaussian process). 

Next, an acquisition function gives the new sampling points (i.e., infill points, :code:`X_new`) based on their usefulness for achieving the optimization goal. 

At this stage, one could choose to visualize the response surfaces using the surrogate model or the infill points locations in the design space. 

A new set of data would be collected by evaluating f at X_new and used to train :math:`\hat{f}`.
This process is repeated until the accuracy of f ̂ is satisfactory or the optima location :math:`{\bf x^{*}} = \underset{{\bf x} \in A}{\operatorname{argmax}} f({\bf x})` is found.


NEXTorch Design
----------------

.. image:: ../_images/user/workflow.svg


As a software package, NEXTorch is structured in a similar way to the active learning framework. 
Initially, users are left to identify the parameters and objectives and frame the optimization problems they work with. 
The key information required includes the ranges and types (categorical, ordinal, continuous, or mixed) of each parameter. 
It would also be helpful to know the sensitivity of parameters by performing exploratory data analysis (such as PCA or random forest).

Depending on the availability of the objective function, NEXTorch supports two types of optimization loop: 

1. automated optimization, where the analytical form of the objective function is known and provide to the software (in the form of a Python object), 
   often in the case of computer simulations
2. human-in-the-loop optimization, where the objective function is unknown, often in the case of laboratory experiments. 

We call the action of generating data from the objective function an “experiment,” which is also the name of the core class in NEXTorch. 

In (1), data are passed through the loop, and experiments are evaluated at the new trials suggested by the acquisition function automatically. 

In (2), visualization could help the users decide whether to carry on the experiments or adjust the experimental setup. 

The users are left to perform the experiments and supply data at the new trials. NEXTorch reads CSV or Excel files from users and exports the data in the same formats. 



BO Implementation Step by Step 
--------------------------------

The major steps of implementing BO in NEXTorch can be summarized as following:

1. Import :code:`nextorch` and other packages
2. Define the objective function and the design space
3. Define the initial sampling plan
4. Initialize an :code:`Experiment` object
5. Run optimization trials
6. Visualize the model reponses
7. Export the optimum

NEXTorch is implemented in a modular fashion which makes running each step much easier. We will go through the main modules and functions next.
