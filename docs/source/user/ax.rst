============
Using Ax 
============

Both Ax_ and NEXTorch are wrappers of BoTorch_ to facilitate Bayesian Optimization for real-world problems. 
BoTorch is the backend engine of all the Bayesian Optimization search and Gaussian Process fitting. 

Ax and NEXTorch define the parameter space, perform parameter normalization/standardization or encoding/decoding, and visualize the results. 
The optimal values obtained from both Ax and NEXTorch should therefore be similar. Their differences lie in syntax and software design.  
As users of Ax since 2019, we feel its syntax and functions are hard to navigate. Ax offers multiple APIs at different levels of automation. 
Neither one is straightforward for human-in-the-loop designs. We find it challenging to generate the next experiment points from acquisition functions without digging into BoTorch functions directly. 
It is also difficult to export the surrogate model predictions and generate one’s own plots. Customizing the initial DOE and the number of samples per iteration is also not found in the documentation. 
As a result, we decided to develop our own BoTorch wrapper, which suits our purposes. Our problems are often in the continuous space with few constraints. 
Needless to say, Ax is an excellent piece of software for adaptive experimentation, including online A/B testing, machine learning model hyperparameter tuning, and other applications in software development. 
However, most of these functions would be left unused in the chemistry and engineering problems. 
The terminology is also different: objectives are called metrics; the parameter space is called the search space; a sample point (a combination of all parameter values) is called an arm, to mention a few. 

To sum up, compared to Ax, NEXTorch offers a more lightweight design, simpler syntax, greater flexibility, and more options for DOE and visualization functions. 
These features allow human-in-the-loop design where decision-making on generating future data is aided by domain knowledge.  

We provide an additional example (Using Ax for optimizating PFR yield) here to highlight the syntax and design differences between Ax and BoTorch. 

.. nbgallery::
    :name: ax-example
    :glob:

    ../examples/12_Using_Ax


.. _Ax: https://ax.dev/
.. _BoTorch: https://botorch.org/
