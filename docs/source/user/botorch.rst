=============================
BoTorch Models and Functions
=============================

NEXTorch supports most acquisition functions and GP models from the upstream BoTorch_.

The default choice of GP model is SingleTaskGP_.

We include EI, PI, UCB, their Monte Carlo variants (qEI, qPI, qUCB), and qEHVI for multi-objective optimization (MOO). 
The `acquisition function objects`_ from BoTorch and their notation in NEXTorch are:

======================================== ================
BoTorch objects                           Notations
======================================== ================
:code:`ExpectedImprovement`               EI
:code:`ProbabilityOfImprovement`          PI
:code:`UpperConfidenceBound`              UCB
:code:`qExpectedImprovement`              qEI
:code:`qProbabilityOfImprovement`         qPI
:code:`qUpperConfidenceBound`             qUCB
:code:`qExpectedHypervolumeImprovement`   qEHVI 
======================================== ================


.. _BoTorch: https://botorch.org/
.. _SingleTaskGP: https://botorch.org/api/models.html#botorch.models.model.Model
.. _`acquisition function objects`: https://botorch.org/api/acquisition.html