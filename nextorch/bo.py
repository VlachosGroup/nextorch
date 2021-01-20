"""
nextorch.bo

Contains Gaussian Processes (GP) and Bayesian Optimization (BO) methods 
"""

import numpy as np
import torch
from torch import Tensor
import copy

from typing import Optional, TypeVar, Union, Tuple, List

# bortorch functions
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound, qProbabilityOfImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_torch

import nextorch.utils as ut
from nextorch.utils import Array, Matrix, ArrayLike1d, MatrixLike2d

# Dictionary for compatiable acqucision functions
acq_dict = {'EI': ExpectedImprovement, 
            'PI': ProbabilityOfImprovement,
            'UCB': UpperConfidenceBound,
            'qEI': qExpectedImprovement, 
            'qPI': qProbabilityOfImprovement,
            'qUCB': qUpperConfidenceBound}
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
    model: bortorch model
        A single task GP, fit to X and Y
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
    model : bortorch model
        a single task GP
    Xs : Tensor
        Independent data, new observation
    Ys : Tensor
        Dependent data, new observation

    Returns
    -------
    model: bortorch model
        A single task GP, fit to X and Y
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

def evaluate_objective_func(
    X_unit: MatrixLike2d, 
    X_range: MatrixLike2d, 
    objective_func: object
) -> Tensor:
    """Evaluate the objective function

    Parameters
    ----------
    X_unit : MatrixLike2d, matrix or 2d tensor
        X in a unit scale
    X_range : MatrixLike2d, matrix or 2d tensor
        list of x ranges
    test_function : function object
        a test function which evaluate np arrays

    Returns
    -------
    Y_tensor: Tensor
        model predicted values
    """
    # Convert matrix type from tensor to numpy matrix
    if isinstance(X_unit, torch.Tensor):
        X_unit_np = X_unit.cpu().numpy()
    else:
        X_unit_np = X_unit.copy()
        
    if isinstance(X_range, torch.Tensor):
        X_range_np = X_range.cpu().numpy()
    else:
        X_range_np = X_range.copy()
    # transform to real scale 
    print(X_range_np)
    X_real = ut.inverse_unitscale_X(X_unit_np, X_range_np)
    # evaluate y
    Y = objective_func(X_real)
    # Convert to tensor
    Y_tensor = torch.tensor(Y, dtype = dtype)

    return Y_tensor

def predict_model(model: Model, X_test: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Makes standardized prediction at X_test using the GP model

    Parameters
    ----------
    model : Model
        A GP model
    X_test : Tensor
        X Tensor used for testing, must have the same dimension 
        as X for training

    Returns
    -------
    Y_test: Tensor
        Standardized prediction, the mean of postierior
    Y_test_lower: Tensor 
        The lower confidence interval 
    Y_test_upper: Tensor
        The upper confidence interval    
    """
    # Make a copy
    X_test = copy.deepcopy(X_test)
    X_test = torch.tensor(X_test, dtype = dtype)
    # Extract postierior distribution
    posterior = model.posterior(X_test)
    Y_test = posterior.mean
    Y_test_lower, Y_test_upper = posterior.mvn.confidence_region()
    
    return Y_test, Y_test_lower, Y_test_upper


def predict_real(model: Model, X_test: Tensor, Y_mean: Tensor, Y_std: Tensor
) -> Tuple[Matrix, Matrix, Matrix]:
    """Make predictions in real scale and returns numpy array

    Parameters
    ----------
    model : Model
        A GP model
    X_test : Tensor
        X Tensor used for testing, must have the same dimension 
        as X for training
    Y_mean : Tensor
        The mean of initial Y set
    Y_std : Tensor
        The std of initial Y set

    Returns
    -------
    Y_test_real: numpy matrix
        predictions in a real scale
    Y_test_lower_real: numpy matrix 
        The lower confidence interval in a real scale
    Y_test_upper_real: numpy matrix 
        The upper confidence interval in a real scale
    """
    # Make standardized predictions using the model
    Y_test, Y_test_lower, Y_test_upper = predict_model(model, X_test)
    # Inverse standardize and convert it to numpy matrix
    Y_test_real = ut.inversestandardize_X(Y_test, Y_mean, Y_std)
    Y_test_real = Y_test_real.detach().numpy()
    
    Y_test_lower_real = ut.inversestandardize_X(Y_test_lower, Y_mean, Y_std)
    Y_test_lower_real = Y_test_lower_real.detach().numpy()
    
    Y_test_upper_real = ut.inversestandardize_X(Y_test_upper, Y_mean, Y_std)
    Y_test_upper_real = Y_test_upper_real.detach().numpy()
    
    return Y_test_real, Y_test_lower_real, Y_test_upper_real
        
def get_acq_func(
        model: Model,
        acq_func_name: str, 
        minmize: Optional[bool] = True, 
        beta: Optional[float] = 0.2,
        best_f: Optional[float] = 1.0,
        **kwargs
) -> AcquisitionFunction:

    err_msg = 'Input acqucision function is not allow. Select from: '
    for ki in acq_dict.keys():
        err_msg += ki + ',' 
    err_msg.split(',')

    if not acq_func_name in acq_dict.keys():
        raise KeyError(err_msg)
    
    acq_object = acq_dict[acq_func_name]

    if acq_func_name == 'EI':
        acq_func = acq_object(model, best_f = best_f, maximize = (not minmize), **kwargs)
    elif acq_func_name == 'PI':
        acq_func = acq_object(model, best_f = best_f, maximize = (not minmize), **kwargs)
    elif acq_func_name == 'UCB':
        acq_func = acq_object(model, beta = beta, maximize = (not minmize), **kwargs)
    elif acq_func_name == 'qEI':
        acq_func = acq_object(model, best_f = best_f, **kwargs)
    elif acq_func_name == 'qPI':
        acq_func = acq_object(model, best_f = best_f, **kwargs)
    else: # acq_func_name == 'qUCB':
        acq_func = acq_object(model, beta = beta, **kwargs)

    return acq_func



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

    

#%%

class Experiment():
    """Experiment object
    Consists of a set of trial points
    Input data is MatrixLike2D
    Data is passed in Tensor"""

    def __init__(self, name: Optional[str] = 'simple_experiment'):
        """Define the name of the epxeriment

        Parameters
        ----------
        name : Optional[str], optional
            Name of the experiment, by default 'simple_experiment'
        """
        self.name = name

    def preprocess_data(self, 
                        X_real: MatrixLike2d,
                        Y_real: MatrixLike2d,
                        preprocessed: Optional[bool] = False,
                        X_ranges: Optional[MatrixLike2d] = None,
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
        preprocessed : Optional[bool], optional
            by default False, the input data will be processed
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

        Raises
        ------
        ValueError
            When inputting X_ranges and unit_flag is True 
        """

        self.X_real = X_real  # independent variable in real unit, numpy array
        self.Y_real = Y_real  # dependent variable in real unit, numpy array

        err_msg = "Variable ranges not needed if they are in a unit scale. \
                    Consider dropping X_ranges input or setting unit_flag as False"

        # Case 1, Input is Tensor, preprocessed
        # X is in a unit scale and 
        # Y is standardized with a zero mean and a unit variance
        if preprocessed: 
            X = self.X_real.detach().clone()
            Y = self.Y_real.detach().clone()
            Y_mean = torch.zeros(self.n_objectives)
            Y_std = torch.ones(self.n_objectives)
            
            if X_ranges is None:
                X_ranges = [[0,1]] * self.n_dim
            else:
                raise ValueError(err_msg)
        
        # Case 2, Input is numpy matrix, not processed
        else: 
            # Set X_ranges
            if X_ranges is None:
                if unit_flag:
                    X_ranges = [[0,1]] * self.n_dim
                else:
                    X_ranges = ut.get_ranges_X(X_real)
            else:
                if unit_flag:
                    raise ValueError(err_msg)

            # Scale X
            X = ut.unitscale_X(X_real, 
                                X_ranges = X_ranges, 
                                unit_flag = unit_flag, 
                                log_flags = log_flags, 
                                decimals = decimals)
            # Standardize Y
            Y = ut.standardize_X(Y_real)
            # Convert to Tensor
            X = torch.tensor(X, dtype=dtype)
            Y = torch.tensor(Y, dtype=dtype)
            # Get mean and std
            Y_mean = torch.tensor(Y_real.mean(axis = 0), dtype = dtype)
            Y_std = torch.tensor(Y_real.std(axis = 0), dtype = dtype)

        # Assign to self
        self.X = X
        self.Y = Y
        self.X_init = X.detach().clone()
        self.Y_init = Y.detach().clone()
        self.X_ranges = X_ranges
        self.Y_mean = Y_mean
        self.Y_std = Y_std


    def input_data(self,
        X_real: MatrixLike2d,
        Y_real: MatrixLike2d,
        Y_weights: Optional[ArrayLike1d] = None,
        X_names: Optional[List[str]] = None,
        Y_names: Optional[List[str]] = None, 
        preprocessed: Optional[bool] = False,
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
        Y_weights: Optional[ArrayLike1d], optional
            Weights of each objectives, sums to 1, used for multi-objetcive function
            by default None, used for single-objective function
        X_names : Optional[List[str]], optional
            Names of independent varibles, by default None
        Y_names : Optional[List[str]], optional
            Names of dependent varibles, by default None
        preprocessed : Optional[bool], optional
            by default False, the input data will be processed
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

        # get specs of the data
        self.n_dim = X_real.shape[1] # number of independent variables
        self.n_points = X_real.shape[0] # number of data points
        self.n_points_init = X_real.shape[0] # number of data points for the initial design
        self.n_objectives = Y_real.shape[1] # number of dependent variables

        # assign variable names
        if X_names is None:
            X_names = ['X' + str(i+1) for i in range(self.n_dim)]
        if Y_names is None:
            Y_names = ['Y' + str(i+1) for i in range(self.n_objectives)]
        self.X_names = X_names
        self.Y_names = Y_names

        # assign weights for objectives 
        if Y_weights is None:
            Y_weights = torch.div(torch.ones(self.n_objectives), self.n_objectives)
        if not isinstance(Y_weights, Tensor):
            Y_weights = torch.tensor(Y_weights, dtype=dtype)
        self.Y_weights = Y_weights

        # Preprocess the data 
        self.preprocess_data(X_real, 
                            Y_real,
                            preprocessed = preprocessed,
                            X_ranges = X_ranges,
                            unit_flag = unit_flag,
                            log_flags = log_flags, 
                            decimals = decimals)
        '''
        Some print statements
        '''

    def update_bestseen(self):

        # Obtain the best value seen so far in Y
        if self.minmize:
            best_values = self.Y.min(dim=0)[0]
        else: 
            best_values = self.Y.max(dim=0)[0]
        # leverage the weights
        best_value_scalar = torch.dot(self.Y_weights, best_values)

        return best_value_scalar

    def set_optim_specs(self,
        objective_func: Optional[object] = None,  
        model: Optional[Model] = None, 
        minmize: Optional[bool] = True,
    ):  
        """Set the specs for Bayseian Optimization

        Parameters
        ----------
        objective_func : Optional[object], by default None
            objective function that we try to optimize
        model : Optional[Model], optional
            pre-trained GP model, by default None
        acq_func : Optional[AcquisitionFunction], optional
            Acqucision function, by default ExpectedImprovement
        minmize : Optional[bool], optional
            by default True, minimize the objective function
            Otherwise False, maximize the objective function
        """
        self.objective_func = objective_func

        if model is None:
            self.model = create_and_fit_gp(self.X, self.Y)

        self.minmize = minmize


    def generate_next_point(self, 
        acq_func_name: Optional[str] = 'EI', 
        n_candidates: Optional[int] = 1,
        beta: Optional[float] = 0.2,
        **kwargs
    ) -> Tuple[Tensor, object]:
        """Generate the next experiment point(s)

        Parameters
        ----------
        n_candidates : Optional[int], optional
            Number of candidate points, by default 1
            The point maximizes the acqucision function

        Returns
        -------
        X_new, acq_func_current: Tuple[Tensor, object]
            X_new: the candidate point matrix 
            acq_func_current: acquction function object
        """
        
        #print(best_value_scalar)
        self.beta = beta
        self.acq_func_name = acq_func_name

        # Update the best_f if necessary
        best_f = None
        if self.acq_func_name in ['EI', 'PI', 'UCB']:
            best_f = self.update_bestseen()
        
        acq_func = get_acq_func(self.model, 
                                self.acq_func_name, 
                                minmize= self.minmize, 
                                beta = self.beta,
                                best_f = best_f,
                                **kwargs)
        
        unit_bounds = torch.stack([torch.zeros(self.n_dim), torch.ones(self.n_dim)])

        #print(unit_bounds)
        X_new, _ = optimize_acqf(acq_func, 
                                bounds= unit_bounds, 
                                q=n_candidates, 
                                num_restarts=10, 
                                raw_samples=100)

        return X_new, acq_func

    def run_trial(self, 
        X_new: Tensor,
        Y_new_real: Optional[Tensor] = None
    ) -> Tensor:
        """Run trial candidate points
        Fit the GP model to new data

        Parameters
        ----------
        X_new: Tensor 
            The new candidate point matrix 
        Y_new_real: Tensor
            Experimental reponse values

        Returns
        -------
        Y_new: Tensor
            values of reponses at the new point values 
        """

        # Case 1, no objective function is specified 
        # Must input Y_new_real
        # Otherwise, raise error
        if self.objective_func is None:
            
            err_msg = "No objective function is specified. The experimental reponse must be provided."
            raise ValueError(err_msg)

        # Case 2, Predict Y_new from objective function
        # Standardize Y_new_real from the prediction
        else:
            Y_new_real = evaluate_objective_func(X_new, self.X_ranges, self.objective_func)
            Y_new = ut.standardize_X(Y_new_real, self.Y_mean, self.Y_std)
                
        # Combine all the training data
        self.X = torch.cat((self.X, X_new))
        self.Y = torch.cat((self.Y, Y_new))
        # Increment the number of points by n_candidate
        self.n_points += X_new.shape[0]
        
        # Add the new point into the model
        self.model = fit_with_new_observations(self.model, X_new, Y_new)
        
        return Y_new

    def predict(self, 
        X_test: Tensor, 
        show_confidence: Optional[bool] = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Use GP model for prediction at X_test

        Parameters
        ----------
        X_test : Tensor
            X Tensor used for testing, must have the same dimension 
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
        Y_test, Y_test_lower, Y_test_upper = predict_model(self.model, X_test)
        if show_confidence:
            return Y_test, Y_test_lower, Y_test_upper
        
        return Y_test

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
        Y_real, Y_lower_real, Y_upper_real = predict_real(self.model, self.X)
        if show_confidence:
            return Y_real, Y_lower_real, Y_upper_real
        
        return Y_real


        
        


    

        





