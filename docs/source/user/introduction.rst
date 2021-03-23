============
Introduction
============


What This Is 
=============

Welcome to NEXTorch! This is an open-source software package in Python/PyTorch to faciliate experimental design using Bayesian Optimization (BO). 
NEXTorch is also a library for learning the theory and implementation of BO. 

Active learning refers to the idea of a machine learning algorithm “learning” from data, proposing next experiments or calculations, 
and improving prediction accuracy with fewer training data or lower computational cost. 
BO, a popular active learning framework, refers to a suite of techniques for global optimization of expensive functions.



Prior works
===============

Machine learning and scientific research communities have developed several BO (kriging) software tools and most of them have interfaces in Python. 
We curate a list of open-source BO packages and provide further discussions in our paper. 

Their documentation pages can be found at:

- Spearmint_
- GyOpt_
- GPflowOpt_
- BoTorch_ and Ax_ 
- Dragonfly_
- Cornell-MOE_
- Emukit_
- sckit-optimize_
- edbo_
- COMBO_
- PyKrige_

Among them, Spearmint_ and GyOpt_ are among the early works to make BO accessible to end users. 
Recently, some packages, such as BoTorch_ and GPflowOpt_, are built on popular machine learning frameworks such as PyTorch_ and TensorFlow_ 
to benefit from the fast matrix operations, batched computation, and GPU acceleration. 
BoTorch stands out since it naturally supports parallel optimization, Monte Carlo acquisition functions, and advanced cases such as multi-task and multi-objective optimization. 
The PyTorch backend also makes it suitable for easy experimentation and fast prototyping. 


Why we build this?
====================

However, most tools are designed for AI researchers or software engineers, often requiring a steep learning curve. The workflow can also be less transparent to end-users. 
Occasionally, design choices are made intentionally to keep humans out of the optimization loop.
The above reasons make them difficult to extend to chemistry or engineering problems, where domain knowledge is essential. 

We have seen attempts being made by the authors of edbo_ (a Bayesian reaction optimization package). 
They performed extensive testing and benchmark studies to showcase the effectiveness of the method in a recent Nature paper `[1]`_.
However, the software is still based on command-line scripts, and clear documentation is lacking. Edbo also has no access to hardware acceleration or the latest state-of-art BO methods.

From a practical perspective, we believe a BO tool should be scalable, flexible, and accessible to the end-users, i.e., chemists and engineers. 
Hence, we build NEXTorch, extending the capabilities of BoTorch, to democratize the use of BO in chemical sciences. 


Why NEXTorch
=============

NEXTorch is unique for several reasons:

1. NEXTorch benefits from the modern architecture and a variety of models, functions offered by BoTorch.

2. NEXTorch provides connections to real-world problems, going beyond BoTorch, including automatic parameter scaling, data type conversions, and visualization capabilities. 
These features allow human-in-the-loop design where decision-making on the next experiments can be aided by domain knowledge.

3. NEXTorch is modular in design which makes it easy to extend to other frameworks. It also serves as a library for learning the theory and implementation of BO. 

We believe its ease of use could serve the community including experimentalists with little or no programming background. 

----------------

Reference: 
--------

`[1]`_ Shields, B. J.; Stevens, J.; Li, J.; Parasram, M.; Damani, F.; Alvarado, J. I. M.; Janey, J. M.; Adams, R. P.; Doyle, A. G. Bayesian Reaction Optimization as a Tool for Chemical Synthesis. Nature 2021, 590, 89–96.


.. _Spearmint: https://github.com/HIPS/Spearmint
.. _GyOpt: https://sheffieldml.github.io/GPyOpt/
.. _GPflowOpt: https://gpflowopt.readthedocs.io/en/latest/intro.html
.. _BoTorch: https://botorch.org/
.. _Ax: https://ax.dev/
.. _Dragonfly: https://dragonfly-opt.readthedocs.io/en/master/
.. _Cornell-MOE: https://github.com/wujian16/Cornell-MOE
.. _Emukit: https://emukit.readthedocs.io/en/latest/#
.. _sckit-optimize: https://scikit-optimize.github.io/stable/
.. _edbo: https://b-shields.github.io/edbo/index.html
.. _COMBO: https://github.com/tsudalab/combo
.. _PyKrige: https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html
.. _PyTorch: https://pytorch.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _[1]: https://www.nature.com/articles/s41586-021-03213-y