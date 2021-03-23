=========
Glossary
=========

acquisition function
^^^^^^^^^^^^^^^^^^^^
A heuristic function to evaluate the usefulness of one more sampling point (i.e., an infill point) for achieving the 
optimization goal. The cost of optimizing the acquisition function is usually negligible compared to optimizing the 
objective function. Often denoted by :math:`\alpha(x)`.

active learning
^^^^^^^^^^^^^^^
A machine learning algorithm which achieves greater accuracy with fewer training data and lower computational cost if 
it is allowed to choose the data from which is learns.

array
^^^^^
A vector, a collection of elements of the same type in a one-dimensional list-like structure.

batching
^^^^^^^^
Creating parallel jobs in computation for acceleration purposes.

Bayes’ theorem
^^^^^^^^^^^^^^
Posterior distribution (:math:`P(A|B)`) is proportional to the likelihood function (:math:`P(B|A)`) multiplied by the prior 
distribution (:math:`P(A)`).

Bayesian Optimization
^^^^^^^^^^^^^^^^^^^^^
A global optimization method. A sequential design strategy to optimize black-box or expensive-to-evaluate functions 
that does not assume any functional forms. The method does not require the first- or second-order derivatives of the 
objective function, referred to as “derivative free.”

Bayesian statistics
^^^^^^^^^^^^^^^^^^^
An approach to data analysis based on Bayes’ theorem, where available knowledge about parameters in a statistical model 
is updated with the information in observed data (or evidence, observations).

black-box function
^^^^^^^^^^^^^^^^^^
A function that we cannot access but we can only observe its output values based on given input values.

categorical variables
^^^^^^^^^^^^^^^^^^^^^
Non-numeric variables that tend to be descriptive, denoted by words, text, or symbols. For examples, 
{catalyst A, catalyst B, catalyst C} or {fertilizer, no fertilizer}.  

continuous variables
^^^^^^^^^^^^^^^^^^^^
Numerical variables that can take any value in a given range.

curse of dimensionality
^^^^^^^^^^^^^^^^^^^^^^^
The phenomena that occur when classifying, organizing, and analyzing high dimensional data that does not occur in low 
dimensional spaces.

design of experiments
^^^^^^^^^^^^^^^^^^^^^
A systematic method to determine the relationship between input variables and output response at the data collection 
stage. Examples include full factorial, central composite, and Latin-Hypercube designs. 

design space
^^^^^^^^^^^^
The multidimensional combination and interaction of input variables, the space composed of :math:`d` input variables, i.e., 
:math:`X\in A\in \mathbb{R}^{d}`, where :math:`d` is the dimensionality of the input.

dimensionality
^^^^^^^^^^^^^^
The number of input variables or parameters, usually noted by :math:`d`. Typically :math:`d \leqslant 20` in most successful 
applications of Bayesian Optimization.

expected improvement
^^^^^^^^^^^^^^^^^^^^
A type of acquisition function, expected improvement over the current best observed value. It is the default choice in NEXTorch. 

experiment
^^^^^^^^^^
Used in a restricted sense to describe those investigations involving the deliberate manipulation of independent variables 
and observing, by measurement, the response of the dependent variables. It can be physical experiments performed in the 
lab or computational experiments. In NEXTorch, an experiment refers to the process of making predictions and searching 
for global optimum for a specific problem via surrogate modeling and Bayesian Optimization.

exploitation
^^^^^^^^^^^^
Evaluating at points with high expected performance near the location where we have seen the previous best point. It is 
valuable because good approximate global optima are likely to reside at such points. 

exploitation-exploration trade-off
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The question of where to sample based on the trade-off between high expected performance and high uncertainty. The choice 
of the acquisition function should balance the trade-off for efficient global optima search. 

exploration
^^^^^^^^^^^
Evaluating at points with high uncertainty. It teaches about the objective functions in locations where we have little 
knowledge, and which tend to be far away from where we have previously sampled. A point that is substantially better 
than one we have seen previously may reside there. 

full factorial design
^^^^^^^^^^^^^^^^^^^^^
A design consists of two or more factors, each with discrete possible values or "levels", and whose samples take on all 
possible combinations of these levels across all such factors.

gaussian process
^^^^^^^^^^^^^^^^
A model that constructs a joint probability distribution over the variables, assuming a multivariate Gaussian distribution. 
The most common surrogate model when using Bayesian Optimization. A GP is specified by its mean function and kernel (covariance) 
function over random variables. The choice of hyperparameters in the model is determine by maximizing the cost function, namely, 
maximum likelihood estimate (MLE).   

global optimization
^^^^^^^^^^^^^^^^^^^
Methods to find the global minima or maxima of an objective function on a given set of input variables. 

global optimum
^^^^^^^^^^^^^^
A feasible solution with a value for the objective function that is as good or better than any other feasible solutions.

heatmap
^^^^^^^
A representation of response values using a color gradient. In NEXTorch, the heatmap plot displays a two-dimensional 
plane showing how the response changes with two input variables while other input variable values are fixed. All points 
that have the same response value are represented by the same color. 

infill point
^^^^^^^^^^^^
A new sample point used to update the surrogate during the process of optimization. We often chose the point where the 
acquisition function is maximized, i.e., :math:`{\bf x_{n+1}}=argmax \alpha_{n}(x)`. 

kernel
^^^^^^
The covariance function in the gaussian processes. Common choices include constant, linear, Radial Basis Function (RBF), 
periodic, Matérn kernels etc. Denoted by :math:`\Sigma(X,X^{'})` over random variables :math:`X` and :math:`X^{'}`.

kriging
^^^^^^^
Originally in geostatistics. A combination of a linear regression model and a zero-mean Gaussian process fitted to the residual 
errors of the linear model. Models are usually fit using a variogram, whereas models are usually fit through maximum likelihood 
in a Gaussian process. Sometimes people use the term Gaussian process or kriging interchangeably. 

Latin hypercube sampling
^^^^^^^^^^^^^^^^^^^^^^^^
A statistical design for generating a near-random sample of factor values from a multidimensional distribution.

level
^^^^^
A (possibly qualitative) value of the “factor” employed in the experimental determination of a response. Often used in 
a DOE. 

likelihood function
^^^^^^^^^^^^^^^^^^^
The conditional probability distribution of the given parameters of the data.

local optimum
^^^^^^^^^^^^^
A feasible solution that is better than neighboring solutions, but does not guarantee to be the best within the entire 
design space.

matrix
^^^^^^
A two-dimensional array with :math:`n` rows and :math:`m` columns.

Monte Carlo
^^^^^^^^^^^
The method of repeated random sampling to approximate deterministic numerical results which are often computational prohibitive 
to evaluate, such as integrals.

multi-fidelity
^^^^^^^^^^^^^^
 
multi-objective optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The method to minimize or maximize multiple objectives. It often involves more than one objective functions that are to 
be minimized or maximized optimized simultaneously. 

multi-task
^^^^^^^^^^

normalization
^^^^^^^^^^^^^
A practice in data preprocessing, also known as min-max scaling, where the range of a variable (or feature) is scaled to 
the [0, 1] interval, i.e., a unit scale. :math:`x_{unit}=(x-min⁡(x))/(max⁡(x)-min⁡(x))`. Bayesian Optimization works well 
with normalized input parameters and therefore they are normalized in NEXTorch.

objective function
^^^^^^^^^^^^^^^^^^
A function of interest to be maximized or minimized. It can be complex computer simulations or real-world experiments. 
The output of the objective function is usually expensive, time-consuming, or otherwise difficult to measure. Denoted 
by :math:`f`, :math:`y=f(x)`. 

objectives
^^^^^^^^^^
Goals of the optimization (maximization or minimization of a certain response variable).

ordinal variables
^^^^^^^^^^^^^^^^^
Numerical variables that take ordered discrete values. For example, integers or float numbers at a fixed interval in a 
given range. 

parameters
^^^^^^^^^^
Input variables to the objective function, or independent variables, or factors in DOE, or features/descriptors in 
machine learning, often denoted by :math:`x_{1},x_{2},...,x_{d}`, or :math:`X` (a :math:`n` by :math:`d` matrix); each 
of :math:`x_{i}` is a vector :math:`x_{i}={(x_{1i},x_{2i},…,x_{ni})}^{T}`; :math:`n` is the number of sample points and 
:math:`d` is the dimensionality of the input. 

pareto front
^^^^^^^^^^^^
The boundary defined by the entire feasible solution set from multi-objective optimization. The optimization algorithm 
faces tradeoff when deriving a set of solutions between the competing objectives, for example, model complexity versus 
model accuracy.

posterior distribution
^^^^^^^^^^^^^^^^^^^^^^
A way to summarize one’s updated knowledge, balancing prior knowledge with observed data, expressed as probability 
distributions.

predictions
^^^^^^^^^^^
Estimated values of the responses given the input values using the surrogate model, denoted by :math:`\hat{f}(x)`. In 
NEXTorch, we use the posterior mean of the gaussian process models as the predictions. We can also report the confidence 
interval. 

prior distribution
^^^^^^^^^^^^^^^^^^
Beliefs about the parameters in a statistical model before seeing the data, expressed as probability distributions. 

probability of improvement
^^^^^^^^^^^^^^^^^^^^^^^^^^
A type of acquisition function, probability of improvement over the current best observed value :math:`f(x_{n^{+}})`, 
defined as :math:`PI(x)=P(f(x) \geqslant f(x_{n^{+}}))`. Here :math:`x_{n^{+}}` is the best point observed so far in a 
set of n points.  

q-acquisition function
^^^^^^^^^^^^^^^^^^^^^^
An acquisition function used in BoTorch (https://botorch.org/docs/acquisition) where (quasi-) Monte-Carlo sampling are 
used to approximate the integrals when evaluating the acquisition function. Examples include qEI, qUCB, qPI etc.

random design
^^^^^^^^^^^^^
A design consists of randomized combinations of factors. The levels of the factors are randomly assigned to samples.

response surface
^^^^^^^^^^^^^^^^
The mathematical relationship between a response variable and input variables. In NEXTorch, the response surface plot 
displays a three-dimensional view showing how the response changes with two input variables while other input variable 
values are fixed.

responses
^^^^^^^^^
Output variables of the objective function, or dependent variables, often denoted by :math:`y_{1},y_{2},...y_{m}`, or 
:math:`Y` (a :math:`n` by :math:`m` matrix), where :math:`n` is the number of sample points and :math:`m` is the number 
is the dimensionality of the output. 

sample
^^^^^^
An observation, a single data point from the objective function. Denoted by :math:`x_{i}`.

sampling plan
^^^^^^^^^^^^^
The spatial arrangement where the observations are built on, :math:`X=\{ x_{1},x_{2},x_{3},...x_{d} \}`. The initial 
sample plans are often generated from a DOE method.

scalar
^^^^^^
A single number.

standardization
^^^^^^^^^^^^^^^
A practice in data preprocessing, also known as Z-score normalization, which ensures the values of each a variable 
(or feature) to have a zero-mean and unit-variance. :math:`x_{standard}=(x-\bar{x})/ \sigma_{x}`. Bayesian Optimization 
works well with standardized responses and therefore the responses are standardized in NEXTorch. 

surrogate model
^^^^^^^^^^^^^^^
A cheap and fast model which is used to approximate the objective function output given a set of input values, also called 
meta-models. Denoted by :math:`\hat{f}`. In Bayesian Optimization, the surrogate model is usually a GP. 

tensor
^^^^^^
A generalized matrix, or multidimensional arrays.

trial
^^^^^
An investigation involving running a batch of experimental points. In NEXTorch, a trial refers to the process of generating 
a set of infill points in each iteration to update the surrogate model.

upper confidence bound
^^^^^^^^^^^^^^^^^^^^^^
A type of acquisition function, that comprises of the posterior mean plus the posterior standard deviation weighted by 
a trade-off parameter, :math:`\beta: UCB(x)=\mu_{n}(x)+\beta \sigma_{n}(x)`. The igher the :math:`\beta` value, the 
higher the amount of exploration.
