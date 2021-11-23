"""
Imports Gaussian Processes (GP) and Bayesian Optimization (BO) methods from BoTorch

Contains all Experiment classes
"""


import os, sys
import subprocess
import numpy as np
import torch
from torch import Tensor
import copy

from typing import Optional, TypeVar, Union, Tuple, List

# bortorch functions
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import AcquisitionObjective, ScalarizedObjective, LinearMCObjective
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound, qProbabilityOfImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_torch

import nextorch.utils as ut
from nextorch.utils import Array, Matrix, ArrayLike1d, MatrixLike2d 
from nextorch.utils import tensor_to_np, np_to_tensor
from nextorch.parameter import Parameter, ParameterSpace

# Dictionary for compatiable acqucision functions
acq_dict = {'EI': ExpectedImprovement, 
            'PI': ProbabilityOfImprovement,
            'UCB': UpperConfidenceBound,
            'qEI': qExpectedImprovement, 
            'qPI': qProbabilityOfImprovement,
            'qUCB': qUpperConfidenceBound,
            'qEHVI': qExpectedHypervolumeImprovement}
"""dict: Keys are the names, values are the BoTorch objects"""


# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dtype = torch.float
torch.set_default_dtype(dtype)

def create_and_fit_gp(X: Tensor, Y: Tensor) -> Model:
    """Creates a GP model to fit the data

    Parameters
    ----------
    X : Tensor
        Independent data
    Y : Tensor
        Depedent data

    Returns
    -------
    model: 'botorch.models.model.Model'_
        A single task GP, fit to X and Y
    
    :_'botorch.models.model.Model': https://botorch.org/api/models.html#botorch.models.model.Model
    """

    #the model is a single task GP
    model = SingleTaskGP(train_X=X, train_Y=Y) 
    model.train()
    # maximize marginal loglikihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_torch(mll)
    mll.eval()
    model.eval()

    return model    

def fit_with_new_observations(model: Model, X: Tensor, Y: Tensor) -> Model:
    """Fit the model with new observation

    Parameters
    ----------
    model : 'botorch.models.model.Model'_
        a single task GP
    Xs : Tensor
        Independent data, new observation
    Ys : Tensor
        Dependent data, new observation

    Returns
    -------
    model: 'botorch.models.model.Model'_
        A single task GP, fit to X and Y

    :_'botorch.models.model.Model': https://botorch.org/api/models.html#botorch.models.model.Model
    """
    # Add the new point into the model
    model = model.condition_on_observations(X=X, Y=Y)
    model.train()
    # maximize marginal loglikihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_torch(mll)
    mll.eval()
    model.eval()
    
    return model

def eval_objective_func(
    X_unit: MatrixLike2d, 
    X_range: MatrixLike2d, 
    objective_func: object,
    return_type: Optional[str] = 'tensor'
) -> MatrixLike2d:
    """Evaluate the objective function

    Parameters
    ----------
    X_unit : MatrixLike2d, matrix or 2d tensor
        X in a unit scale
    X_range : MatrixLike2d, matrix or 2d tensor
        list of x ranges
    objective_func : function object
        a objective function to optimize
    return_type: Optional[str], optional
        either 'tensor' or 'np'
        by default 'tensor'

    Returns
    -------
    Y: MatrixLike2d
        model predicted values

    Raises
    ------
    ValueError
        if input return_type not defined
    """
    # Convert to tensor
    if return_type not in ['tensor', 'np']:
        raise ValueError('return_type must be either tensor or np')

    # Convert matrix type from tensor to numpy matrix
    X_unit_np = tensor_to_np(X_unit)
    #X_range_np = tensor_to_np(X_range)    
    
    # transform to real scale 
    X_real = ut.inverse_unitscale_X(X_unit_np, X_range)
    # evaluate y
    Y = objective_func(X_real)

    if return_type == 'tensor':
        Y = np_to_tensor(Y)

    return Y

def eval_objective_func_encoding(
    X_unit: MatrixLike2d, 
    parameter_space: MatrixLike2d, 
    objective_func: object,
    return_type: Optional[str] = 'tensor'
) -> MatrixLike2d:
    """Evaluate the objective function

    Parameters
    ----------
    X_unit : MatrixLike2d, matrix or 2d tensor
        X in a unit scale
    X_range : MatrixLike2d, matrix or 2d tensor
        list of x ranges
    objective_func : function object
        a objective function to optimize
    return_type: Optional[str], optional
        either 'tensor' or 'np'
        by default 'tensor'

    Returns
    -------
    Y: MatrixLike2d
        model predicted values

    Raises
    ------
    ValueError
        if input return_type not defined
    """
    # Convert to tensor
    if return_type not in ['tensor', 'np']:
        raise ValueError('return_type must be either tensor or np')

    # Convert matrix type from tensor to numpy matrix
    X_unit_np = tensor_to_np(X_unit)
    #X_range_np = tensor_to_np(X_range)    
    
    # transform to real scale 
    X_real = ut.encode_to_real_ParameterSpace(X_unit_np, parameter_space)
    # evaluate y
    Y = objective_func(X_real)

    if return_type == 'tensor':
        Y = np_to_tensor(Y)

    return Y



def model_predict(
    model: Model, 
    X_test: MatrixLike2d,
    return_type: Optional[str] = 'tensor',
    negate_Y: Optional[bool] = False 
) -> Tuple[MatrixLike2d, MatrixLike2d, MatrixLike2d]:
    """Makes standardized prediction at X_test using the GP model

    Parameters
    ----------
    model : 'botorch.models.model.Model'
        A GP model
    X_test : MatrixLike2d
        X matrix used for testing, must have the same dimension 
        as X for training
    return_type: Optional[str], optional
        either 'tensor' or 'np'
        by default 'tensor'
    negate_Y: Optional[bool], optional
        if true, negate the model predicted values 
        in the case of minimization
        by default False

    Returns
    -------
    Y_test: MatrixLike2d
        Standardized prediction, the mean of postierior
    Y_test_lower: MatrixLike2d 
        The lower confidence interval 
    Y_test_upper: MatrixLike2d
        The upper confidence interval 

    Raises
    ------
    ValueError
        if input return_type not defined 

    :_'botorch.models.model.Model': https://botorch.org/api/models.html#botorch.models.model.Model
    """
    if return_type not in ['tensor', 'np']:
        raise ValueError('return_type must be either tensor or np')

    # Make a copy
    X_test = np_to_tensor(X_test)

    # Extract postierior distribution
    posterior = model.posterior(X_test)
    Y_test = posterior.mean
    Y_test_lower, Y_test_upper = posterior.mvn.confidence_region()

    if negate_Y:
        Y_test = -1*Y_test
        Y_test_lower = -1*Y_test_lower
        Y_test_upper = -1*Y_test_upper
        
    if return_type == 'np':
        Y_test = tensor_to_np(Y_test), 
        Y_test_lower = tensor_to_np(Y_test_lower)
        Y_test_upper = tensor_to_np(Y_test_upper)

    return Y_test, Y_test_lower, Y_test_upper


def model_predict_real(
    model: Model, 
    X_test: MatrixLike2d, 
    Y_mean: MatrixLike2d, 
    Y_std: MatrixLike2d,
    return_type: Optional[str] = 'tensor',
    negate_Y: Optional[bool] = False 
) -> Tuple[MatrixLike2d, MatrixLike2d, MatrixLike2d]:
    """Make predictions in real scale and returns numpy array

    Parameters
    ----------
    model : 'botorch.models.model.Model'_
        A GP model
    X_test : MatrixLike2d
        X matrix used for testing, must have the same dimension 
        as X for training
    Y_mean : MatrixLike2d
        The mean of initial Y set
    Y_std : MatrixLike2d
        The std of initial Y set
    return_type: Optional[str], optional
        either 'tensor' or 'np'
        by default 'tensor'
    negate_Y: Optional[bool], optional
        if true, negate the model predicted values 
        in the case of minimization
        by default False

    Returns
    -------
    Y_test_real: numpy matrix
        predictions in a real scale
    Y_test_lower_real: numpy matrix 
        The lower confidence interval in a real scale
    Y_test_upper_real: numpy matrix 
        The upper confidence interval in a real scale
    
    :_'botorch.models.model.Model': https://botorch.org/api/models.html#botorch.models.model.Model
    """
    if return_type not in ['tensor', 'np']:
        raise ValueError('return_type must be either tensor or np')

    # Make standardized predictions using the model
    Y_test, Y_test_lower, Y_test_upper = model_predict(model, X_test, negate_Y=negate_Y)
    # Inverse standardize and convert it to numpy matrix
    Y_test_real = ut.inverse_standardize_X(Y_test, Y_mean, Y_std)
    Y_test_lower_real = ut.inverse_standardize_X(Y_test_lower, Y_mean, Y_std)
    Y_test_upper_real = ut.inverse_standardize_X(Y_test_upper, Y_mean, Y_std)

    if return_type == 'np':
        Y_test_real = tensor_to_np(Y_test_real)
        Y_test_lower_real = tensor_to_np(Y_test_lower_real)
        Y_test_upper_real = tensor_to_np(Y_test_upper_real)
    
    return Y_test_real, Y_test_lower_real, Y_test_upper_real


# def predict_mesh_2D(model, X_test,  mesh_size, Y_mean, Y_std):
#     '''
#     Predict 2d mesh values from surrogates 
#     Generate plots if required
#     Return outputs Y in 2d numpy matrix 
#     '''
#     # predict the mean for ff model, returns a standardized 1d tensor
#     Y_test, _, _ = ut.predict_model(model, X_test)
#     # Inverse the standardization and convert 1d y into a 2d array
#     Y_test_real = ut.transform_mesh_2D_Y(Y_test, Y_mean, Y_std, mesh_size)
    
#     return Y_test_real


def get_acq_func(
        model: Model,
        acq_func_name: str, 
        beta: Optional[float] = 0.2,
        best_f: Optional[float] = 1.0,
        objective: Optional[AcquisitionObjective] = None, 
        **kwargs
) -> AcquisitionFunction:
    """Get a specific type of acqucision function

    Parameters
    ----------
    model : 'botorch.models.model.Model'_
        A GP model
    acq_func_name : str
        Name of the acquisition function
        Must be one of "EI", "PI", "UCB", "qEI", "qPI", "qUCB"
    beta : Optional[float], optional
        hyperparameter used in UCB, by default 0.2
    best_f : Optional[float], optional
        best value seen so far used in PI and EI, by default 1.0
    objective : Optional['botorch.acquisition.objective.AcquisitionObjective'_], optional
        Linear objective constructed from a weight vector,
        Used for multi-ojective optimization

    **kwargs：keyword arguments
        Other parameters used by 'botorch.acquisition'_

    Returns
    -------
    acq_func: 'botorch.acquisition.AcquisitionFunction'_
        acquisition function object
        
    Raises
    ------
    KeyError
        if input name is not a validate acquisition function

    :_'botorch.models.model.Model': https://botorch.org/api/models.html#botorch.models.model.Model
    :_'botorch.acquisition.objective.AcquisitionObjective': https://botorch.org/api/acquisition.html#botorch.acquisition.objective.AcquisitionObjective
    :_'botorch.acquisition': https://botorch.org/api/acquisition.html
    :_'botorch.acquisition.AcquisitionFunction': https://botorch.org/api/acquisition.html#botorch.acquisition.acquisition.AcquisitionFunction
    
    """

    err_msg = 'Input acquisition function is not allow. Select from: '
    for ki in acq_dict.keys():
        err_msg += ki + ',' 
    err_msg.split(',')
    # check if the name input is valid
    if not acq_func_name in acq_dict.keys():
        raise KeyError(err_msg)
    # get the object
    acq_object = acq_dict[acq_func_name]
    # input key parameters
    if acq_func_name == 'EI':
        acq_func = acq_object(model, best_f=best_f, objective=objective, **kwargs)
    elif acq_func_name == 'PI':
        acq_func = acq_object(model, best_f=best_f, objective=objective, **kwargs)
    elif acq_func_name == 'UCB':
        acq_func = acq_object(model, beta=beta, objective=objective, **kwargs)
    elif acq_func_name == 'qEI':
        acq_func = acq_object(model, best_f=best_f, objective=objective,**kwargs)
    elif acq_func_name == 'qPI':
        acq_func = acq_object(model, best_f=best_f, objective=objective,**kwargs)
    elif acq_func_name == 'qUCB':
        acq_func = acq_object(model, beta=beta, objective=objective, **kwargs)
    else: # acq_func_name == 'qEHVI':
        acq_func = acq_object(model, **kwargs)

    return acq_func

def eval_acq_func(
    acq_func: AcquisitionFunction, 
    X_test: MatrixLike2d,
    return_type: Optional[str] = 'tensor'
) -> MatrixLike2d:
    """Evaluate acquisition function at test values

    Parameters
    ----------
    acq_func : 'botorch.acquisition.AcquisitionFunction'_
        acquisition function object
    X_test : MatrixLike2d
        X matrix used for testing, must have the same dimension 
        as X for training
    return_type: Optional[str], optional
        either 'tensor' or 'np'
        by default 'tensor'

    Returns
    -------
    acq_val_test: MatrixLike2d
        acquisition function value at X_test
        
    Raises
    ------
    ValueError
        if input return_type not defined 

    .._'botorch.acquisition.AcquisitionFunction': https://botorch.org/api/acquisition.html
    """
    if return_type not in ['tensor', 'np']:
        raise ValueError('return_type must be either tensor or np')

    X_test = np_to_tensor(X_test)
    n_dim = 1
    # compute acquicision function values at X_test and X_train
    # the input needs to be formatted as a 3D tensor
    acq_val_test = acq_func(X_test.view((X_test.shape[0],1, n_dim)))
    if return_type == 'np':
        acq_val_test = tensor_to_np(acq_val_test)

    return acq_val_test


def get_top_k_candidates(
    acq_func: AcquisitionFunction, 
    acq_func_name: str, 
    bounds: Tensor,
    k: Optional[int] = 1
) -> Tensor:
    """Return the top k candidates which 
    maximize the acqusicition function value

    Parameters
    ----------
    acq_func : 'botorch.acquisition.AcquisitionFunction'_
        acquisition function object
    acq_func_name : str
        Name of the acquisition function
        Must be one of "EI", "PI", "UCB", "qEI", "qPI", "qUCB", "qEHVI"
    bounds : Tensor
        Bounds of each X
    k : Optional[int], optional
        number of candidates, by default 1

    Returns
    -------
    X_new: Tensor
        Top k candidate points, shape of n_dim * k

    .._'botorch.acquisition.AcquisitionFunction': https://botorch.org/api/acquisition.html
    """

    return_best_only = True
    # Case 1 - if a Monte Carlo acquisition function is used
    # set the batch size equal to k, return the best of each batch
    if acq_func_name in ['qEI', 'qPI', 'qUCB', 'qEHVI']:
        X_new, acq_value = optimize_acqf(acq_func, 
                                        bounds= bounds, 
                                        q=k, 
                                        num_restarts=10, 
                                        raw_samples=100, 
                                        return_best_only=return_best_only,
                                        sequential=True)

    # Case 2 - if an analytical acquisition function is used
    # return the best k points based on the acquisition values
    else:
        if k > 1:  return_best_only = False
        X_new, acq_value = optimize_acqf(acq_func, 
                            bounds= bounds, 
                            q=1,  # q must be 1 for analytical 
                            num_restarts=10, 
                            raw_samples=100, 
                            return_best_only=return_best_only)

        if k > 1:
            indices_top_k = torch.topk(acq_value.view(-1), k=k, dim=0).indices
            X_new = X_new[indices_top_k].squeeze(-1)

    return X_new
    


#%%
class Database():
    """
    Database class
    The base class for the experiment classes
    Handles data input and directory setup
    """
    def __init__(self, name: Optional[str] = 'simple_experiment'):
        """Define the name of the epxeriment

        Parameters
        ----------
        name : Optional[str], optional
            Name of the experiment, by default 'simple_experiment'
        """
        self.name = name
        self.parameter_space = None

        # Set up the path to save graphical results
        parent_dir = os.getcwd()
        exp_path = os.path.join(parent_dir, self.name)
        self.exp_path = exp_path


    def define_space(self, parameters: Union[Parameter, List[Parameter]]):
        """Define the parameter space

        Parameters
        ----------
        parameters : Union[Parameter, List[Parameter]]
            A single or a list of parameters 
        """
        self.parameter_space = ParameterSpace(parameters)

        # Assign properties to self
        self.X_ranges = self.parameter_space.X_ranges # 2d list
        self.all_continous = self.parameter_space.all_continuous

    def encode_X(self, X_unit: MatrixLike2d) -> Tuple[MatrixLike2d, MatrixLike2d]:
        """Encode X from the relax continuous space
        in a unit scale to the encoding space

        Parameters
        ----------
        X_unit : MatrixLike2d
            original X in a unit scale

        Returns
        -------
        X_encode: MatrixLike2d
            Encoded X in a unit scale
        X_real: MatrixLike2d
            Encoded X in a real scale
        """        
        # Encoding simply finds the nearest point in the continous space 
        X_encode = ut.unit_to_encode_ParameterSpace(X_unit, 
                                            parameter_space=self.parameter_space,
                                            log_flags=self.log_flags, 
                                            decimals=self.decimals)
        # Get X_real from X_encode
        X_real = ut.encode_to_real_ParameterSpace(X_encode, 
                                            parameter_space=self.parameter_space,
                                            log_flags=self.log_flags, 
                                            decimals=self.decimals)

        return X_encode, X_real


    def preprocess_data(self, 
                        X_real: MatrixLike2d,
                        Y_real: MatrixLike2d,
                        standardized: Optional[bool] = False,
                        unit_flag: Optional[bool] = False,
                        log_flags: Optional[list] = None, 
                        decimals: Optional[int] = None
    ):
        """Preprocesses input data and assigns the variables to self

        Parameters
        ----------
        X_real : MatrixLike2d
            original independent data in a real scale
        Y_real : MatrixLike2d
            original dependent data in a real scale
        standardized : Optional[bool], optional
            by default False, the input Y is standardized 
            if true, skip processing
        X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
        unit_flag: Optional[bool], optional,
            by default, False 
            If true, the X is in a unit scale (from DOE) 
        log_flags : Optional[list], optional
            list of boolean flags
            True: use the log scale on this dimensional
            False: use the normal scale 
            by default []
        decimals : Optional[int], optional
            Number of decimal places to keep
            by default None, i.e. no rounding up

        """
        X_real = tensor_to_np(X_real)
        Y_real = tensor_to_np(Y_real)
        

        # Step 1, standardize Y
        # if Y is standardized with a zero mean and a unit variance
        if standardized: 
            Y = np_to_tensor(Y_real)  
            Y_mean = torch.zeros(self.n_objectives)
            Y_std = torch.ones(self.n_objectives)
        # Standardize Y
        else:
            Y = ut.standardize_X(Y_real)
            # Get mean and std
            Y_mean = np_to_tensor(Y_real.mean(axis = 0))
            Y_std = np_to_tensor(Y_real.std(axis = 0))
            

        # Step 2, encode X if all_continuous is false
        # X is the input X_real
        X = X_real.copy()
        # if input X is in a unit scale, an DOE plan for example
        if unit_flag:    
            if not self.all_continous:
                # Encode X
                X, X_real = self.encode_X(X)
        # if input X is in a real scale
        else:
            # Scale X to a unit scale
            if not self.all_continous:
                X = ut.real_to_encode_ParameterSpace(X, 
                                                    parameter_space=self.parameter_space,
                                                    log_flags=log_flags, 
                                                    decimals=decimals)
            else:
                X = ut.unitscale_X(X, 
                                X_ranges = self.X_ranges, 
                                log_flags = log_flags, 
                                decimals = decimals)

        # Step 3, Compute X_real from X 
        # Compute X_real from X unit if all_continuous is true 
        # and input X is in a unit scale
        if self.all_continous and unit_flag:
            X_real = ut.inverse_unitscale_X(X, 
                                            X_ranges = self.X_ranges, 
                                            log_flags = log_flags, 
                                            decimals = decimals)
        
        # Convert to Tensor
        X = np_to_tensor(X)
        Y = np_to_tensor(Y)
            
        # Assign to self
        self.X = X # in a unit scale, tensor
        self.Y = Y # in a unit scale, tensor
        self.X_init = X.detach().clone() # in a unit scale, tensor
        self.Y_init = Y.detach().clone() # in a unit scale, tensor
        self.Y_mean = Y_mean # tensor
        self.Y_std = Y_std # tensor

        self.X_real = X_real  # in a real scale, numpy array
        self.Y_real = Y_real  # in a real scale, numpy array
        self.X_init_real = X_real.copy()  # in a real scale, numpy array
        self.Y_init_real = Y_real.copy()  # in a real scale, numpy array


    def input_data(self,
        X_real: MatrixLike2d,
        Y_real: MatrixLike2d,
        X_names: Optional[List[str]] = None,
        Y_names: Optional[List[str]] = None, 
        standardized: Optional[bool] = False,
        X_ranges: Optional[MatrixLike2d] = None,
        unit_flag: Optional[bool] = False,
        log_flags: Optional[list] = None, 
        decimals: Optional[int] = None
    ):
        """Input data into Experiment object
        
        Parameters
        ----------
        X_real : MatrixLike2d
            original independent data in a real scale
        Y_real : MatrixLike2d
            original dependent data in a real scale
        X_names : Optional[List[str]], optional
            Names of independent varibles, by default None
        Y_names : Optional[List[str]], optional
            Names of dependent varibles, by default None
        standardized : Optional[bool], optional
            by default False, the input Y is standardized 
            if true, skip processing
        X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
        unit_flag: Optional[bool], optional,
            by default, False 
            If true, the X is in a unit scale so
            the function is used to scale X to a log scale
        log_flags : Optional[list], optional
            list of boolean flags
            True: use the log scale on this dimensional
            False: use the normal scale 
            by default []
        decimals : Optional[int], optional
            Number of decimal places to keep
            by default None, i.e. no rounding up
        """
        # expand to 2D
        if len(X_real.shape)<2:
            X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
        if len(Y_real.shape)<2:
            Y_real = np.expand_dims(Y_real, axis=1) #If 1D, make it 2D array

        # get specs of the data
        self.n_dim = X_real.shape[1] # number of independent variables
        self.n_points = X_real.shape[0] # number of data points
        self.n_points_init = X_real.shape[0] # number of data points for the initial design
        self.n_objectives = Y_real.shape[1] # number of dependent variables

        # Define the default parameter space for all continuous variable
        if self.parameter_space is None:
            parameters = [Parameter() for _ in range(self.n_dim)]
            # Set the X ranges for the parameters
            if X_ranges is not None:
                X_ranges = ut.expand_ranges_X(X_ranges) # 2d list
                for i in range(self.n_dim):
                    parameters[i].x_range = X_ranges[i]    
            self.define_space(parameters)

        # Assign variable names
        if X_names is None:
            X_names = []
            # Pull names from ParameterSpace first
            for i, x_name_i in enumerate(self.parameter_space.names):
                if x_name_i is None:
                    if self.n_dim == 1:
                        X_names = ['x']
                    else:
                        X_names.append('x' + str(i+1))
                else:
                    X_names.append(x_name_i)

        if Y_names is None:
            if self.n_objectives > 1:
                Y_names = ['y' + str(i+1) for i in range(self.n_objectives)]
            else:
                Y_names = ['y']

        self.X_names = X_names
        self.Y_names = Y_names

        # Set other specs
        self.log_flags = log_flags # list of bool
        self.decimals =decimals # list of int

        # Preprocess the data 
        self.preprocess_data(X_real, 
                            Y_real,
                            standardized=standardized,
                            unit_flag=unit_flag,
                            log_flags=log_flags, 
                            decimals=decimals)


# #%%
# class Database():
#     """
#     Database class
#     The base class for the experiment classes
#     Handles data input and directory setup
#     """
#     def __init__(self, name: Optional[str] = 'simple_experiment'):
#         """Define the name of the epxeriment

#         Parameters
#         ----------
#         name : Optional[str], optional
#             Name of the experiment, by default 'simple_experiment'
#         """
#         self.name = name

#         # Set up the path to save graphical results
#         parent_dir = os.getcwd()
#         exp_path = os.path.join(parent_dir, self.name)
#         self.exp_path = exp_path

#     def preprocess_data(self, 
#                         X_real: MatrixLike2d,
#                         Y_real: MatrixLike2d,
#                         preprocessed: Optional[bool] = False,
#                         X_ranges: Optional[MatrixLike2d] = None,
#                         unit_flag: Optional[bool] = False,
#                         log_flags: Optional[list] = None, 
#                         decimals: Optional[int] = None
#     ):
#         """Preprocesses input data and assigns the variables to self

#         Parameters
#         ----------
#         X_real : MatrixLike2d
#             original independent data in a real scale
#         Y_real : MatrixLike2d
#             original dependent data in a real scale
#         preprocessed : Optional[bool], optional
#             by default False, the input data will be processed
#             if true, skip processing
#         X_ranges : Optional[MatrixLike2d], optional
#             list of x ranges, by default None
#         unit_flag: Optional[bool], optional,
#             by default, False 
#             If true, the X is in a unit scale (from DOE) 
#         log_flags : Optional[list], optional
#             list of boolean flags
#             True: use the log scale on this dimensional
#             False: use the normal scale 
#             by default []
#         decimals : Optional[int], optional
#             Number of decimal places to keep
#             by default None, i.e. no rounding up

#         """
#         X_real = tensor_to_np(X_real)
#         Y_real = tensor_to_np(Y_real)
        
#         # Case 1, Input is Tensor, preprocessed
#         # X is in a unit scale and 
#         # Y is standardized with a zero mean and a unit variance
#         if preprocessed: 
#             X = np_to_tensor(self.X_real)  
#             Y = np_to_tensor(self.Y_real)  
#             Y_mean = torch.zeros(self.n_objectives)
#             Y_std = torch.ones(self.n_objectives)
#             # Update X_ranges
#             if X_ranges is None:
#                 X_ranges = [[0,1]] * self.n_dim

#             #  X_real needs to be computed 
#             X_real = ut.inverse_unitscale_X(X_real, 
#                                             X_ranges = X_ranges, 
#                                             unit_flag = unit_flag, 
#                                             log_flags = log_flags, 
#                                             decimals = decimals)
        
#         # Case 2, Input is numpy matrix, not processed
#         # Handle X first than Y
#         else: 
#             # Case 2.1, Input X is in a unit scale, an DOE plan for example
#             if unit_flag:
#                 # X is the input X_real
#                 X = X_real.copy()
#                 # Set X_ranges
#                 if X_ranges is None:
#                     X_ranges = [[0,1]] * self.n_dim    
#                 # X_real needs to be computed
#                 X_real = ut.inverse_unitscale_X(X_real, 
#                                                 X_ranges = X_ranges, 
#                                                 log_flags = log_flags, 
#                                                 decimals = decimals)
#             # Case 2.2 Input X is in a real scale
#             else:
#                 # Set X_ranges
#                 if X_ranges is None:
#                     X_ranges = ut.get_ranges_X(X_real)
                    
#                 # Scale X to a unit scale
#                 X = ut.unitscale_X(X_real, 
#                                    X_ranges = X_ranges, 
#                                    log_flags = log_flags, 
#                                    decimals = decimals)

#             # Standardize Y
#             Y = ut.standardize_X(Y_real)
#             # Convert to Tensor
#             X = np_to_tensor(X)
#             Y = np_to_tensor(Y)
#             # Get mean and std
#             Y_mean = np_to_tensor(Y_real.mean(axis = 0))
#             Y_std = np_to_tensor(Y_real.std(axis = 0))

#         # Assign to self
#         self.X = X # in a unit scale, tensor
#         self.Y = Y # in a unit scale, tensor
#         self.X_init = X.detach().clone() # in a unit scale, tensor
#         self.Y_init = Y.detach().clone() # in a unit scale, tensor
#         self.X_ranges = ut.expand_ranges_X(X_ranges) # 2d list
#         self.Y_mean = Y_mean # tensor
#         self.Y_std = Y_std # tensor

#         self.log_flags = log_flags # list of bool
#         self.decimals =decimals # list of int

#         self.X_real = X_real  # in a real scale, numpy array
#         self.Y_real = Y_real  # in a real scale, numpy array
#         self.X_init_real = X_real.copy()  # in a real scale, numpy array
#         self.Y_init_real = Y_real.copy()  # in a real scale, numpy array


#     def input_data(self,
#         X_real: MatrixLike2d,
#         Y_real: MatrixLike2d,
#         X_names: Optional[List[str]] = None,
#         Y_names: Optional[List[str]] = None, 
#         preprocessed: Optional[bool] = False,
#         X_ranges: Optional[MatrixLike2d] = None,
#         unit_flag: Optional[bool] = False,
#         log_flags: Optional[list] = None, 
#         decimals: Optional[int] = None
#     ):
#         """Input data into Experiment object
        
#         Parameters
#         ----------
#         X_real : MatrixLike2d
#             original independent data in a real scale
#         Y_real : MatrixLike2d
#             original dependent data in a real scale
#         X_names : Optional[List[str]], optional
#             Names of independent varibles, by default None
#         Y_names : Optional[List[str]], optional
#             Names of dependent varibles, by default None
#         preprocessed : Optional[bool], optional
#             by default False, the input data will be processed
#             if true, skip processing
#         X_ranges : Optional[MatrixLike2d], optional
#             list of x ranges, by default None
#         unit_flag: Optional[bool], optional,
#             by default, False 
#             If true, the X is in a unit scale so
#             the function is used to scale X to a log scale
#         log_flags : Optional[list], optional
#             list of boolean flags
#             True: use the log scale on this dimensional
#             False: use the normal scale 
#             by default []
#         decimals : Optional[int], optional
#             Number of decimal places to keep
#             by default None, i.e. no rounding up
#         """
#         # expand to 2D
#         if len(X_real.shape)<2:
#             X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
#         if len(Y_real.shape)<2:
#             Y_real = np.expand_dims(Y_real, axis=1) #If 1D, make it 2D array

#         # get specs of the data
#         self.n_dim = X_real.shape[1] # number of independent variables
#         self.n_points = X_real.shape[0] # number of data points
#         self.n_points_init = X_real.shape[0] # number of data points for the initial design
#         self.n_objectives = Y_real.shape[1] # number of dependent variables

#         # assign variable names
#         if X_names is None:
#             if self.n_dim > 1:
#                 X_names = ['x' + str(i+1) for i in range(self.n_dim)]
#             else:
#                 X_names = ['x']
#         if Y_names is None:
#             if self.n_objectives > 1:
#                 Y_names = ['y' + str(i+1) for i in range(self.n_objectives)]
#             else:
#                 Y_names = ['y']
#         self.X_names = X_names
#         self.Y_names = Y_names

#         # Preprocess the data 
#         self.preprocess_data(X_real, 
#                             Y_real,
#                             preprocessed = preprocessed,
#                             X_ranges = X_ranges,
#                             unit_flag = unit_flag,
#                             log_flags = log_flags, 
#                             decimals = decimals)
#         '''
#         Some print statements
#         '''


#%%
class BasicExperiment(Database):
    """
    BasicExperiment class
    Base: Database
    The generic class for the experiment classes
    Handles prediction and running the next trial
    """
    def fit_model(self):
        """Train a GP model given X, Y and the sign of Y
        """

        # create a GP model based on input data
        # In the case of minimize, the negative reponses values are used to fit the GP
        self.model = create_and_fit_gp(self.X, self.objective_sign * self.Y)
    

    def assign_weights(
        self, 
        Y_weights: Optional[ArrayLike1d] = None):
        """Assign weights to each objective Y

        Parameters
        ----------
        Y_weights : Optional[ArrayLike1d], optional
            Weights assigned to each objective Y, sums to 1
            by default None, each objective is treated equally
        """
        
        # if no input, assign equal weights to each objective 
        if Y_weights is None:
            Y_weights = torch.ones(self.n_objectives)

        # Normalize the weights to sum as 1
        Y_weights = np_to_tensor(Y_weights)
        Y_weights = torch.div(Y_weights, torch.sum(Y_weights))

        self.Y_weights = Y_weights


    def set_optim_specs(self,
        objective_func: Optional[object] = None,  
        model: Optional[Model] = None, 
        maximize: Optional[bool] = True,
        Y_weights: Optional[ArrayLike1d] = None
    ):  
        """Set the specs for Bayseian Optimization

        Parameters
        ----------
        objective_func : Optional[object], by default None
            objective function that is being optimized
        model : Optional['botorch.models.model.Model'_], optional
            pre-trained GP model, by default None
        maximize : Optional[bool], optional
            by default True, maximize the objective function
            Otherwise False, minimize the objective function
        Y_weights : Optional[ArrayLike1d], optional
            Weights assigned to each objective Y, sums to 1
            by default None, each objective is treated equally
        
        :_'botorch.models.model.Model': https://botorch.org/api/models.html#botorch.models.model.Model
        """
        # assign objective function
        self.objective_func = objective_func

        # Set the sign for the objective
        if maximize: 
            self.objective_sign = 1 # sign for the reponses
            self.negate_Y = False # if true (minimization), negate the model predicted values        
        else:
            self.objective_sign = -1 
            self.negate_Y = True
        
        # set optimization goal
        self.maximize = maximize

        # fit a GP model if no model is input
        if model is None:
            self.fit_model()

        # assign weights to each objective, useful only to multi-objective systems
        if Y_weights is not None:
            self.assign_weights(Y_weights)


    def run_trial(self, 
        X_new: MatrixLike2d,
        X_new_real: Matrix,
        Y_new_real: Optional[Matrix] = None,
    ) -> Tensor:
        """Run trial candidate points
        Fit the GP model to new data

        Parameters
        ----------
        X_new: MatrixLike2d 
            The new candidate point matrix 
        X_new_real: Matrix 
            The new candidate point matrix in a real scale
        Y_new_real: Matrix
            Experimental reponse values

        Returns
        -------
        Y_new: Tensor
            values of reponses at the new point values 
        """
        X_new = np_to_tensor(X_new)
        # Case 1, no objective function is specified 
        # Must input Y_new_real
        # Otherwise, raise error
        if Y_new_real is None:
            if self.objective_func is None:
                err_msg = "No objective function is specified. The experimental reponse must be provided."
                raise ValueError(err_msg)

        # Case 2, Predict Y_new from objective function
        # Standardize Y_new_real from the prediction
            else:
                Y_new_real = eval_objective_func(X_new, self.X_ranges, self.objective_func)
        
        # Case 3, Y_new_real is provided from the experiment
        Y_new = ut.standardize_X(Y_new_real, self.Y_mean, self.Y_std)

        # Combine all the training data
        self.X = torch.cat((self.X, X_new))
        self.Y = torch.cat((self.Y, Y_new))
        self.X_real = np.concatenate((self.X_real, X_new_real))
        self.Y_real = np.concatenate((self.Y_real, Y_new_real))

        # Increment the number of points by n_candidates
        self.n_points += X_new.shape[0]
        
        # Add the new point into the model
        self.model = fit_with_new_observations(self.model, X_new, self.objective_sign * Y_new)
        
        return Y_new


    def predict(self, 
        X_test: MatrixLike2d, 
        show_confidence: Optional[bool] = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Use GP model for prediction at X_test

        Parameters
        ----------
        X_test : MatrixLike2d
            X matrix in a unit scale used for testing,
            must have the same dimension 
            as X for training
        show_confidence : Optional[bool], optional
            by default False, only return posterior mean
            If True, return the mean, and lower, upper confidence interval

        Returns
        -------
        Y_test: Tensor
            The mean of postierior 
        Y_test_lower: Tensor, optional
            The lower confidence interval 
        Y_test_upper: Tensor, optional
            The upper confidence interval   
        """
        Y_test, Y_test_lower, Y_test_upper = model_predict(self.model, X_test, negate_Y=self.negate_Y)

        if show_confidence:
            return Y_test, Y_test_lower, Y_test_upper
        
        return Y_test


    def predict_real(self, 
                    X_test: MatrixLike2d,
                    show_confidence: Optional[bool] = False
    ) -> Union[Matrix, Tuple[Matrix, Matrix, Matrix]]:
        """Use GP model for prediction at X_test

        Parameters
        ----------
        X_test : MatrixLike2d
            X matrix in a real scale used for testing, 
            must have the same dimension 
            as X for training
        show_confidence : Optional[bool], optional
            by default False, only return posterior mean
            If True, return the mean, and lower, upper confidence interval

        Returns
        -------
        Y_test_real: numpy matrix
            predictions in a real scale
        Y_test_lower_real: numpy matrix 
            The lower confidence interval in a real scale
        Y_test_upper_real: numpy matrix 
            The upper confidence interval in a real scale
        """
        X_test_real = ut.unitscale_X(X_test, self.X_ranges)
        Y_real, Y_lower_real, Y_upper_real = model_predict_real(self.model, 
                                                                X_test_real, 
                                                                self.Y_mean, 
                                                                self.Y_std, 
                                                                return_type='np',
                                                                negate_Y=self.negate_Y)
        if show_confidence:
            return Y_real, Y_lower_real, Y_upper_real
        
        return Y_real


    def validate_training(self, show_confidence: Optional[bool] = False
    ) -> Union[Matrix, Tuple[Matrix, Matrix, Matrix]]:
        """Use GP model for prediction at training X
        Y_real is used to compare with Y from objective function

        Parameters
        ----------
        show_confidence : Optional[bool], optional
            by default False, only return posterior mean
            If True, return the mean, and lower, upper confidence interval

        Returns
        -------
        Y_test_real: numpy matrix
            predictions in a real scale
        Y_test_lower_real: numpy matrix 
            The lower confidence interval in a real scale
        Y_test_upper_real: numpy matrix 
            The upper confidence interval in a real scale
        """
        Y_real, Y_lower_real, Y_upper_real = model_predict_real(self.model, 
                                                                self.X, 
                                                                self.Y_mean, 
                                                                self.Y_std, 
                                                                return_type='np', 
                                                                negate_Y=self.negate_Y)
        if show_confidence:
            return Y_real, Y_lower_real, Y_upper_real
        
        return Y_real


    def run_trials_auto(self, 
                        n_trials: int,
                        acq_func_name: Optional[str] = 'EI'):
        """Automated optimization loop with one 
        infill point added per loop
        When objective function is defined,
        responses are obtained from the objective function
        otherwise, from the GP model 

        Parameters
        ----------
        n_trials : int
            Number of optimization loops
            one infill point is added per loop
        acq_func_name : Optional[str], optional
            Name of the acquisition function
            Must be one of "EI", "PI", "UCB", "qEI", "qPI", "qUCB"
            by default 'EI'
        """        
        for _ in range(n_trials):
            # Generate the next experiment point
            # by default 1 point per trial
            X_new, X_new_real, _ = self.generate_next_point(acq_func_name)

            # Case 1, no objective function is specified
            # Use GP models as ground truth  
            if self.objective_func is None:
                Y_new_real = self.predict_real(X_new_real, show_confidence=False)
            # Case 2, objective function is specified
            else:
                Y_new_real = self.objective_func(X_new_real)
                
            # Retrain the model by input the next point into Exp object
            self.run_trial(X_new, X_new_real, Y_new_real)



class Experiment(BasicExperiment):
    """
    Experiment class
    Base: BasicExperiment
    Experiment consists of a set of trial points
    For single objective optimization
    """

    def update_bestseen(self) -> Tensor:
        """Calculate the best seen value in Y 

        Returns
        -------
        best_value_scalar: Tensor
            a scalar saved in Tensor object
        """
        # In the case of minimize, the negative reponses values are used in the GP
        best_values = torch.max(self.Y * self.objective_sign, dim=0)[0] 
        best_value_scalar = best_values

        return best_value_scalar


    def generate_next_point(self, 
        acq_func_name: Optional[str] = 'EI', 
        n_candidates: Optional[int] = 1,
        beta: Optional[float] = 0.2,
        **kwargs
    ) -> Tuple[Tensor, Matrix, AcquisitionFunction]:
        """Generate the next trial point(s)

        Parameters
        ----------
        acq_func_name : Optional[str], optional
            Name of the acquisition function
            Must be one of "EI", "PI", "UCB", "qEI", "qPI", "qUCB"
            by default 'EI'
        n_candidates : Optional[int], optional
            Number of candidate points, by default 1
            The point maximizes the acqucision function
        beta : Optional[float], optional
            hyperparameter used in UCB, by default 0.2
        **kwargs：keyword arguments
            Other parameters used by 'botorch.acquisition'_

        Returns
        -------
        X_new: Tensor
            where the acquisition function is optimized
            A new trial shall be run at this point
        X_new: Matrix
            The new point in a real scale
        acq_func: AcquisitionFunction
            Current acquisition function, can be used for plotting

        .._'botorch.acquisition': https://botorch.org/api/acquisition.html
        """
        self.acq_func_name = acq_func_name
        self.beta = beta
        # Update the best_f if necessary
        best_f = None
        if self.acq_func_name in ['EI', 'PI', 'qEI', 'qPI']:
            best_f = self.update_bestseen()

        # Set parameters for acquisition function
        acq_func = get_acq_func(self.model, 
                                self.acq_func_name, 
                                beta = self.beta,
                                best_f = best_f,
                                **kwargs)
        
        unit_bounds = torch.stack([torch.zeros(self.n_dim), torch.ones(self.n_dim)])
        
        # Optimize the acquisition using the default setup
        X_new = get_top_k_candidates(acq_func=acq_func,
                                     acq_func_name=acq_func_name,
                                     bounds=unit_bounds,
                                     k=n_candidates)
        
        # Encode the X_new if all_continous is false
        # Encoding simply finds the nearest point in the continous space 
        # and use it as X_new
        if not self.all_continous:  
            X_new, X_new_real = self.encode_X(X_new)
        else: 
            X_new_real = ut.inverse_unitscale_X(X_new, 
                                                X_ranges = self.X_ranges, 
                                                log_flags= self.log_flags,
                                                decimals = self.decimals)

        # Assign acq_func object to self
        self.acq_func_current = acq_func

        return X_new, X_new_real, acq_func

    
    def get_optim(self) -> Tuple[float, ArrayLike1d, int]:
        """Get the optimal response and conditions 
        from the model

        Returns
        -------
        y_real_opt: float
            Optimal response
        X_real_opt: ArrayLike1d
            parameters or independent variable values 
            at the optimal poinrt
        index_opt: int
            Index of optimal point, zero indexing
        """
        tol = 1e-6 #tolerance for finding match
        # Use np.ufunc.accumulate to find the bestseen min/max
        if self.maximize:
            y_real_opt_accum = np.maximum.accumulate(self.Y_real) 
            y_real_opt = np.max(y_real_opt_accum)
        else:
            y_real_opt_accum = np.minimum.accumulate(self.Y_real) 
            y_real_opt = np.min(y_real_opt_accum)
            
        # find a matching optimum, get the first seen (min) index
        index_opt = np.min(np.where(np.abs(self.Y_real - y_real_opt) < tol)[0])
        # Extract X
        X_real_opt = self.X_real[index_opt]

        return y_real_opt, X_real_opt, index_opt



#%% MOO classes
class SingleWeightedExperiment(BasicExperiment):
    """
    WeightedExperiment class
    Base: BasicExperiment
    Experiments with weighted objectives
    """
    def update_bestseen(self) -> Tensor:
        """Calculate the best seen value in Y 

        Returns
        -------
        best_value_scalar: Tensor
            a scalar saved in Tensor object
        """
        # In the case of minimize, the negative reponses values are used in the GP
        best_values = torch.max(self.Y * self.objective_sign, dim=0)[0]
        # leverage the weights  
        best_value_scalar = torch.dot(self.Y_weights, best_values)

        return best_value_scalar


    def generate_next_point(self, 
        acq_func_name: Optional[str] = 'EI', 
        n_candidates: Optional[int] = 1,
        beta: Optional[float] = 0.2,
        **kwargs
    ) -> Tuple[Tensor, Matrix, AcquisitionFunction]:
        """Generate the next trial point(s)

        Parameters
        ----------
        acq_func_name : Optional[str], optional
            Name of the acquisition function
            Must be one of "EI", "PI", "UCB", "qEI", "qPI", "qUCB"
            by default 'EI'
        n_candidates : Optional[int], optional
            Number of candidate points, by default 1
            The point maximizes the acqucision function
        beta : Optional[float], optional
            hyperparameter used in UCB, by default 0.2
        **kwargs：keyword arguments
            Other parameters used by 'botorch.acquisition'_

        Returns
        -------
        X_new: Tensor
            where the acquisition function is optimized
            A new trial shall be run at this point
        X_new_real: Matrix
            The new point in a real scale
        acq_func: AcquisitionFunction
            Current acquisition function, can be used for plotting

        .._'botorch.acquisition': https://botorch.org/api/acquisition.html
        """
        self.acq_func_name = acq_func_name
        self.beta = beta

        # Update the best_f for analytic acq funcs
        # Set acqucision objective; 
        # ScalarizedObjective for analytic
        # LinearMCObjective for MC
        best_f = None
        if self.acq_func_name in ['EI', 'PI', 'qEI', 'qPI']:
            best_f = self.update_bestseen()
            acq_objective = ScalarizedObjective(weights=self.Y_weights) 
        else:
            acq_objective = LinearMCObjective(weights=self.Y_weights)
        
        
        # Set parameters for acquisition function
        acq_func = get_acq_func(self.model, 
                                self.acq_func_name, 
                                beta = self.beta,
                                best_f = best_f,
                                objective=acq_objective,
                                **kwargs)
        
        unit_bounds = torch.stack([torch.zeros(self.n_dim), torch.ones(self.n_dim)])
        
        # Optimize the acquisition using the default setup
        X_new = get_top_k_candidates(acq_func=acq_func,
                                     acq_func_name=acq_func_name,
                                     bounds=unit_bounds,
                                     k=n_candidates)

        # Encode the X_new if all_continous is false
        # Encoding simply finds the nearest point in the continous space 
        # and use it as X_new
        if not self.all_continous:  
            X_new, X_new_real = self.encode_X(X_new)
        else: 
            X_new_real = ut.inverse_unitscale_X(X_new, 
                                                X_ranges = self.X_ranges, 
                                                log_flags= self.log_flags,
                                                decimals = self.decimals)

        # Assign acq_func object to self
        self.acq_func_current = acq_func

        return X_new, X_new_real, acq_func

    
    def get_weighted_optim(self) -> Tuple[float, ArrayLike1d, int]:
        """Get the weighted optimal response and conditions 
        from the model

        Returns
        -------
        y_real_opt: float
            Optimal response
        X_real_opt: ArrayLike1d
            parameters or independent variable values 
            at the optimal point
        index_opt: int
            Index of optimal point, zero indexing
        """
        Y_weights = tensor_to_np(self.Y_weights)
        y_real_linear = np.dot(self.Y_real, Y_weights)

        tol = 1e-6 #tolerance for finding match
        # Use np.ufunc.accumulate to find the bestseen min/max
        if self.maximize:
            y_real_opt_accum = np.maximum.accumulate(y_real_linear) 
            y_real_opt = np.max(y_real_opt_accum)
        else:
            y_real_opt_accum = np.minimum.accumulate(y_real_linear) 
            y_real_opt = np.min(y_real_opt_accum)
            
        # find a matching optimum, get the first seen (min) index
        index_opt = np.min(np.where(np.abs(y_real_linear - y_real_opt) < tol)[0])
        # Extract X
        X_real_opt = self.X_real[index_opt]
        # Extract Y
        Y_real_opt = self.Y_real[index_opt]

        return Y_real_opt, X_real_opt, index_opt



class WeightedMOOExperiment(Database):
    """
    MOOExperiment class
    Base: Database
    For multi-objective optimization (MOO)
    Currently, only supports two objectives 
    Used for generating Pareto front
    """
    def __init__(self, name: Optional[str] = 'MOO_experiment'):
        """Define the name of the epxeriment

        Parameters
        ----------
        name : Optional[str], optional
            Name of the experiment, by default 'MOO_experiment'
        """
        BasicExperiment.__init__(self, name)


    def set_optim_specs(self,
        weights: Union[ArrayLike1d, float],
        objective_func: Optional[object] = None,  
        maximize: Optional[bool] = True,
    ):  
        """Set the specs for Pareto front Optimization

        Parameters
        ----------
        weights : ArrayLike1d
            List of weights for objective 1 between 0 and 1
        objective_func : Optional[object], by default None
            objective function that is being optimized
        maximize : Optional[bool], optional
            by default True, maximize the objective function
            Otherwise False, minimize the objective function
        
        :_'botorch.models.model.Model': https://botorch.org/api/models.html#botorch.models.model.Model
        """
        self.objective_func = objective_func
        self.maximize = maximize
        if maximize: 
            self.objective_sign = 1
        else:
            self.objective_sign = -1

        # Total number of experiments
        if isinstance(weights, float):
            weights = [weights]
        self.n_exp = len(weights)

        # Compute the weight pairs 
        # The weights for objective 2 is 1-weight_i
        weight_pairs = []
        for weight_i in weights:
            weight_pairs.append([weight_i, 1-weight_i])
        
        # List of experiment objects
        experiments = [] 

        print('Initializing {} experiments'.format(self.n_exp))
        # initialize weighted experimnets with data and weights 
        for i, weight_pair_i in enumerate(weight_pairs):
            experiment_i = SingleWeightedExperiment()
            experiment_i.input_data(self.X_init, 
                                    self.Y_init_real, 
                                    X_ranges = self.X_ranges, 
                                    unit_flag=True)
            experiment_i.set_optim_specs(objective_func=objective_func,
                                         model=None, #start fresh
                                         maximize=maximize, 
                                         Y_weights = weight_pair_i)
            experiments.append(experiment_i)
            print('Initializing experiments {:.2f} % '.format((i+1)/self.n_exp *100))

        self.experiments = experiments
        

    def run_exp_auto(self, 
                     n_trials: int,
                     acq_func_name: Optional[str] = 'EI'
    )-> MatrixLike2d: 
        """run each experiments with Bayesian Optimization
        Extract the optimum points of each experiment

        Parameters
        ----------
        n_trials : int
            Number of optimization loops
            one infill point is added per loop
        acq_func_name : Optional[str], optional
            Name of the acquisition function
            Must be one of "EI", "PI", "UCB", "qEI", "qPI", "qUCB"
            by default 'EI'

        Returns
        -------
        Y_real_opts: MatrixLike2d
            Optimum values given each weight pair
        """

        X_real_opts = [] # optimum locations, in a unit scale
        Y_real_opts = []  # optimum values, in a real scale

        print('Running {} experiments'.format(self.n_exp))
        # train the weighted experiments one by one 
        for i, experiment_i in enumerate(self.experiments):
            # Generate the next experiment point for experiment i 
            # by default 1 point per trial
            experiment_i.run_trials_auto(n_trials, acq_func_name)

            # Save the optimum locations and values
            Y_real_opt, X_real_opt, _ = experiment_i.get_weighted_optim()
            X_real_opts.append(X_real_opt)
            Y_real_opts.append(Y_real_opt)
            print('Running experiments {:.2f} % '.format((i+1)/self.n_exp *100))

        self.X_real_opts = np.array(X_real_opts)
        self.Y_real_opts = np.array(Y_real_opts)

    def get_optim(self) -> Tuple[Matrix, Matrix]:
        """Get the optimal Pareto set

        Returns
        -------
        y_real_opt: Matrix
            set of optimal response at each weight combination
        X_real_opt: Matrix
            set of parameters or independent variable values 
            at each weight combination
        """
        return self.Y_real_opts, self.X_real_opts


class EHVIMOOExperiment(Experiment):
    """
    EHVIMOOExperiment class
    Base: Experiment
    For multi-objective optimization (MOO)
    """

    def set_ref_point(self, 
        ref_point: ArrayLike1d
    ):
        """Set the reference point

        Parameters
        ----------
        ref_point : ArrayLike1d
            reference point for the objectives

        """
        self.ref_point = ref_point


    def generate_next_point(self, 
        acq_func_name: Optional[str] = 'qEHVI', 
        n_candidates: Optional[int] = 1,
        eta: Optional[float] = 0.001,
        **kwargs
    ) -> Tuple[Tensor, Matrix, AcquisitionFunction]:
        """Generate the next trial point(s) using qEHVI

        Parameters
        ----------
        acq_func_name : Optional[str], optional
            Name of the acquisition function
            by default 'qEHVI'
        n_candidates : Optional[int], optional
            Number of candidate points, by default 1
            The point maximizes the acqucision function
        eta : Optional[float], optional
            hyperparameter used in qEHVI, by default 0.001
        **kwargs：keyword arguments
            Other parameters used by 'botorch.acquisition'_

        Returns
        -------
        X_new: Tensor
            where the acquisition function is optimized
            A new trial shall be run at this point
        X_new: Matrix
            The new point in a real scale
        acq_func: AcquisitionFunction
            Current acquisition function, can be used for plotting

        .._'botorch.acquisition': https://botorch.org/api/acquisition.html
        """
        # Set acq func name and hyperparameter
        self.acq_func_name = acq_func_name
        self.eta = eta

        # Set parameters for acquisition function
        partitioning = NondominatedPartitioning(torch.tensor(self.ref_point), Y=self.Y)
        acq_func = get_acq_func(self.model,
                                self.acq_func_name,  
                                ref_point=self.ref_point,
                                partitioning=partitioning, 
                                eta=self.eta,
                                **kwargs)
        
        unit_bounds = torch.stack([torch.zeros(self.n_dim), torch.ones(self.n_dim)])
        
        # Optimize the acquisition using the default setup
        X_new = get_top_k_candidates(acq_func=acq_func,
                                     acq_func_name=acq_func_name,
                                     bounds=unit_bounds,
                                     k=n_candidates)
        
        # Encode the X_new if all_continous is false
        # Encoding simply finds the nearest point in the continous space 
        # and use it as X_new
        if not self.all_continous:  
            X_new, X_new_real = self.encode_X(X_new)
        else: 
            X_new_real = ut.inverse_unitscale_X(X_new, 
                                                X_ranges = self.X_ranges, 
                                                log_flags= self.log_flags,
                                                decimals = self.decimals)

        # Assign acq_func object to self
        self.acq_func_current = acq_func

        return X_new, X_new_real, acq_func

    def get_optim(self) -> Tuple[ArrayLike1d, ArrayLike1d]:
        """Get the optimal response and conditions 
        from the model

        Returns
        -------
        y_real_opt: ArrayLike1d
            Optimal response
        X_real_opt: ArrayLike1d
            parameters or independent variable values 
            at the optimal poinrt
        """

        pareto_mask = is_non_dominated(torch.tensor(self.Y_real))
        Y_real_opt = self.Y_real[pareto_mask]
        X_real_opt = self.X_real[pareto_mask]

        return Y_real_opt, X_real_opt



#%% CFD classes
class COMSOLExperiment(Experiment):
    """
    COMSOLExperiment class
    Base: Experiment
    Experiment consists of a set of trial points using COMSOL
    For single objective optimization
    """

    def input_data(self,
        X_real: MatrixLike2d,
        Y_real: MatrixLike2d,
        X_names: List[str],
        X_units: List[str],
        Y_names: Optional[List[str]] = None,
        Y_units: Optional[List[str]] = None, 
        standardized: Optional[bool] = False,
        X_ranges: Optional[MatrixLike2d] = None,
        unit_flag: Optional[bool] = False,
        log_flags: Optional[list] = None, 
        decimals: Optional[int] = None
    ):
        """Input data into Experiment object
        
        Parameters
        ----------
        X_real : MatrixLike2d
            original independent data in a real scale
        Y_real : MatrixLike2d
            original dependent data in a real scale
        X_names : List[str]
            Names of independent varibles
        X_units : List[str]
            Units of independent varibles
        Y_names : Optional[List[str]]
            Names of dependent varibles
        Y_units : Optional[List[str]]
            Units of dependent varibles            
        standardized : Optional[bool], optional
            by default False, the input Y is standardized 
            if true, skip processing
        X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
        unit_flag: Optional[bool], optional,
            by default, False 
            If true, the X is in a unit scale so
            the function is used to scale X to a log scale
        log_flags : Optional[list], optional
            list of boolean flags
            True: use the log scale on this dimensional
            False: use the normal scale 
            by default []
        decimals : Optional[int], optional
            Number of decimal places to keep
            by default None, i.e. no rounding up
        """

        super().input_data(X_real=X_real,
                        Y_real=Y_real,
                        X_names=X_names,
                        Y_names=Y_names,
                        standardized=standardized,
                        X_ranges=X_ranges,
                        unit_flag=unit_flag,
                        log_flags=log_flags, 
                        decimals=decimals)

        # assign variable names and units
        self.X_names = X_names
        self.X_units = X_units

    def comsol_simulation(self, X_new_real):
        """Run COMSOL simulation
        
        Parameters
        ----------
        X_new_real : MatrixLike2d
            The new point in a real scale
        """

        # update parameters
        for i in range(len(self.X_names)):
            match = '"'+self.X_names[i]+'", "'+str(np.round(self.X_real[-1,i], decimals=8))+'['+self.X_units[i]+']"'
            replace = '"'+self.X_names[i]+'", "'+str(np.round(X_new_real[-1,i], decimals=8))+'['+self.X_units[i]+']"'

            with open(self.objective_file_name+".java","r") as f:
                data = f.read().replace(match,replace)

            with open(self.objective_file_name+".java","w") as f:
                f.write(data)
    
        # run simulations
        subprocess.run([self.comsol_location,  "compile", self.objective_file_name+".java"])
        print("COMSOL file is sucessfully compiled. Simulation starts.")

        process = subprocess.Popen([self.comsol_location,  "batch", "-inputfile", self.objective_file_name+".class"], stdout=subprocess.PIPE)
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line.rstrip())

        print("Simulation is done.")

        # read output objective
        data = np.loadtxt(self.comsol_output_location, skiprows=5, delimiter=',')

        if (data.ndim == 1):
            Y_new_real = np.array([[data[self.comsol_output_col - 1]]])
        else:
            Y_new_real = np.array([[data[-1, self.comsol_output_col - 1]]])

        return Y_new_real

    def set_optim_specs(self, 
        objective_file_name: str,
        comsol_location: str,
        comsol_output_location: str,
        comsol_output_col: Optional[int] = 2, 
        model: Optional[Model] = None, 
        maximize: Optional[bool] = True,
        Y_weights: Optional[ArrayLike1d] = None
    ):  
        """Set the specs for Bayseian Optimization

        Parameters
        ----------
        objective_file_name : str
            the objective COMSOL file
        comsol_location : str
            the location COMSOL installed
        comsol_output_location : str
            the location of saved COMSOL output
            should be a text file
        comsol_output_col : int
            the column number of the objective
        model : Optional['botorch.models.model.Model'_], optional
            pre-trained GP model, by default None
        maximize : Optional[bool], optional
            by default True, maximize the objective function
            Otherwise False, minimize the objective function
        Y_weights : Optional[ArrayLike1d], optional
            Weights assigned to each objective Y, sums to 1
            by default None, each objective is treated equally
        
        :_'botorch.models.model.Model': https://botorch.org/api/models.html#botorch.models.model.Model
        """

        # assign objective COMSOL file and location
        self.objective_file_name = objective_file_name
        self.comsol_location = comsol_location
        self.objective_func = self.comsol_simulation

        # assign output file and objective column
        self.comsol_output_location = comsol_output_location
        self.comsol_output_col = comsol_output_col
        
        # set optimization goal
        self.maximize = maximize

        if maximize: 
            self.objective_sign = 1 # sign for the reponses
            self.negate_Y = False # if true (minimization), negate the model predicted values        
        else:
            self.objective_sign = -1 
            self.negate_Y = True

        # create a GP model based on input data
        # In the case of minimize, the negative reponses values are used to fit the GP
        if model is None:
            self.model = create_and_fit_gp(self.X, self.objective_sign * self.Y)
        # assign weights to each objective, useful only to multi-objective systems
        if Y_weights is not None:
            self.assign_weights(Y_weights)


class COMSOLMOOExperiment(EHVIMOOExperiment):
    """
    COMSOLMOOExperiment class
    Base: EHVIMOOExperiment
    For multi-objective optimization (MOO)
    Used for generating Pareto front
    """

    def input_data(self,
        X_real: MatrixLike2d,
        Y_real: MatrixLike2d,
        X_names: List[str],
        X_units: List[str],
        Y_names: Optional[List[str]] = None,
        Y_units: Optional[List[str]] = None, 
        standardized: Optional[bool] = False,
        X_ranges: Optional[MatrixLike2d] = None,
        unit_flag: Optional[bool] = False,
        log_flags: Optional[list] = None, 
        decimals: Optional[int] = None
    ):
        """Input data into Experiment object
        
        Parameters
        ----------
        X_real : MatrixLike2d
            original independent data in a real scale
        Y_real : MatrixLike2d
            original dependent data in a real scale
        X_names : Optional[List[str]]
            Names of independent varibles
        X_units : Optional[List[str]]
            Units of independent varibles
        Y_names : Optional[List[str]]
            Names of dependent varibles
        Y_units : Optional[List[str]]
            Units of dependent varibles            
        standardized : Optional[bool], optional
            by default False, the input data will be standardized
            if true, skip processing
        X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
        unit_flag: Optional[bool], optional,
            by default, False 
            If true, the X is in a unit scale so
            the function is used to scale X to a log scale
        log_flags : Optional[list], optional
            list of boolean flags
            True: use the log scale on this dimensional
            False: use the normal scale 
            by default []
        decimals : Optional[int], optional
            Number of decimal places to keep
            by default None, i.e. no rounding up
        """

        super().input_data(X_real, Y_real, X_names, Y_names, standardized, X_ranges, unit_flag, log_flags, decimals)

        # assign variable names and units
        self.X_names = X_names
        self.X_units = X_units
     
    def comsol_simulation(self, X_new_real):
        """Run COMSOL simulation
        
        Parameters
        ----------
        X_new_real : MatrixLike2d
            The new point in a real scale
        """

        # update parameters
        for i in range(len(self.X_names)):
            match = '"'+self.X_names[i]+'", "'+str(np.round(self.X_real[-1,i], decimals=8))+'['+self.X_units[i]+']"'
            replace = '"'+self.X_names[i]+'", "'+str(np.round(X_new_real[-1,i], decimals=8))+'['+self.X_units[i]+']"'

            with open(self.objective_file_name+".java","r") as f:
                data = f.read().replace(match,replace)

            with open(self.objective_file_name+".java","w") as f:
                f.write(data)
    
        # run simulations
        subprocess.run([self.comsol_location,  "compile", self.objective_file_name+".java"])
        print("COMSOL file is sucessfully compiled. Simulation starts.")

        process = subprocess.Popen([self.comsol_location,  "batch", "-inputfile", self.objective_file_name+".class"], stdout=subprocess.PIPE)
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line.rstrip())

        print("Simulation is done.")

        # read output objective
        data = np.loadtxt(self.comsol_output_location, skiprows=5, delimiter=',')

        if (data.ndim == 1):
            Y_new_real = np.array([[data[self.comsol_output_col[0] - 1], data[self.comsol_output_col[1] - 1]]])
        else:
            Y_new_real = np.array([[data[-1, self.comsol_output_col[0] - 1], data[-1, self.comsol_output_col[1] - 1]]])

        return Y_new_real

    def set_optim_specs(self, 
        objective_file_name: str,
        comsol_location: str,
        comsol_output_location: str,
        comsol_output_col: Optional[List[int]] = [2, 3], 
        model: Optional[Model] = None, 
        maximize: Optional[bool] = True,
        Y_weights: Optional[ArrayLike1d] = None
    ):  
        """Set the specs for Bayseian Optimization

        Parameters
        ----------
        objective_file_name : str
            the objective COMSOL file
        comsol_location : str
            the location COMSOL installed
        comsol_output_location : str
            the location of saved COMSOL output
            should be a text file
        comsol_output_col : List[int]
            the column number of the objective
        model : Optional['botorch.models.model.Model'_], optional
            pre-trained GP model, by default None
        maximize : Optional[bool], optional
            by default True, maximize the objective function
            Otherwise False, minimize the objective function
        Y_weights : Optional[ArrayLike1d], optional
            Weights assigned to each objective Y, sums to 1
            by default None, each objective is treated equally
        
        :_'botorch.models.model.Model': https://botorch.org/api/models.html#botorch.models.model.Model
        """

        # assign objective COMSOL file and location
        self.objective_file_name = objective_file_name
        self.comsol_location = comsol_location
        self.objective_func = self.comsol_simulation

        # assign output file and objective column
        self.comsol_output_location = comsol_output_location
        self.comsol_output_col = comsol_output_col
        
        # set optimization goal
        self.maximize = maximize

        if maximize: 
            self.objective_sign = 1 # sign for the reponses
            self.negate_Y = False # if true (minimization), negate the model predicted values        
        else:
            self.objective_sign = -1 
            self.negate_Y = True

        # create a GP model based on input data
        # In the case of minimize, the negative reponses values are used to fit the GP
        if model is None:
            self.model = create_and_fit_gp(self.X, self.objective_sign * self.Y)
        # assign weights to each objective, useful only to multi-objective systems
        if Y_weights is not None:
            self.assign_weights(Y_weights)
        
