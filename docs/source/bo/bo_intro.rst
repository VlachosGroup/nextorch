===================
Key Concepts in BO
===================



.. contents:: Table of Contents
    :depth: 2


What can BO do?
===============


Key Concepts and Terminology
============================


Surrogate module
----------------


Gaussian Process
-----------------


Acquisition Functions
---------------------
The acquisition function is applied to obtain the new sampling point **X_new**. It measures the value of evaluating 
the objective function at **X_new**, based on the current posterior distribution over \hat{f}. The most commonly used 
acquisition function is expected improvement (EI). The EI is the expectation taken under the posterior distribution \hat{f} of 
the improvement of the new observation at **X** over the current best observation f^*:



Aside from the EI, there are also other acquisition functions for single objective BO available in NEXTorch, including 
probability of improvement (PI), upper confidence bound (UCB), and their Monte Carlo variants (qEI, qPI, qUCB).



Multi-objective Optimization (MOO)
----------------------------------


