"""
nextorch.experiment

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
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_torch

import nextorch.utils as ut

from typing import Optional, TypeVar, Union, Tuple
# NEED TO EXPLAIN THESE IN DOCS
# Create a type variable for 1D arrays from numpy, np.ndarray
Array = TypeVar('Array')
# Create a type variable for 2D arrays from numpy, np.ndarray, and call it as a matrix
Matrix = TypeVar('Matrix')

# Create a type variable which is array like (1D) including list, array, 1d tensor
ArrayLike1d = Union[list, Array, Tensor]
# Create a type variable which is matrix like (2D) including matrix, tensor, 2d list
# This also includes ArrayList1d types
MatrixLike2d = Union[list, Matrix, Tensor]


# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dtype = torch.float


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
    y_tensor: Tensor
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
    X_real = ut.inverse_unitscale_X(X_unit_np, X_range_np)
    # evaluate y
    y = objective_func(X_real)
    # Convert to tensor
    y_tensor = torch.tensor(y, dtype = dtype)

    return y_tensor

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

    def ___init___(self,
        X_real: MatrixLike2d,
        Y_real: MatrixLike2d,
        preprocessed: Optional[bool] = False,
        X_ranges: Optional[MatrixLike2d] = None,
        unit_flag: Optional[bool] = False,
        log_flags: Optional[list] = None, 
        decimals: Optional[int] = None,
        name: Optional[str] = 'Experiment', 
        X_names: Optional[List[str]] = None,
        Y_names: Optional[List[str]] = None, 
    ):
        """Input data into Experiment object
        
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
        name : Optional[str], optional
            Name of the experiment, by default 'Experiment'
        X_names : Optional[List[str]], optional
            Names of independent varibles, by default None
        Y_names : Optional[List[str]], optional
            Names of dependent varibles, by default None
        
        """
        # Assign variables and names
        self.X_real = X_real
        self.Y_real = Y_real

        if X_ranges is None:
            X_ranges = ut.get_ranges_X(X_real)
        self.X_ranges = X_ranges

        self.n_dim = X_real.shape[1] # number of independent variables
        self.n_points = X_real.shape[0] # number of data points
        self.n_points_init = X_real.shape[0]
        self.n_objectives = Y_real.shape[1] # number of dependent variables

        self.name = name

        if X_names is None:
            X_names = ['X' + str(i+1) for i in range(self.n_dim)]

        if Y_names is None:
            Y_names = ['Y' + str(i+1) for i in range(self.n_objectives)]

        self.X_names = X_names
        self.Y_names = Y_names

        # Case 1, Input is Tensor, preprocessed
        # X is in a unit scale and 
        # Y is standardized with a zero mean and a unit variance
        if preprocessed: 
            X = X_real.detach().clone()
            Y = Y_real.detach().clone()
            Y_mean = torch.zeros(self.n_objectives)
            Y_std = torch.ones(self.n_objectives)
        
        # Case 2, Input is numpy matrix, not processed
        else: 
            X = ut.unitscale_X(X_real, 
                                X_ranges = X_ranges, 
                                unit_flag = unit_flag, 
                                log_flags = log_flags, 
                                decimals = decimals)

            Y = ut.standardize_X(Y_real)
            # Convert to Tensor
            X = torch.tensor(X)
            Y = torch.tensor(Y)
            Y_mean = Y.mean(axis = 0).detach().clone()
            Y_std = Y.std(axis = 0).detach().clone()

        # Assign to self
        self.X = X
        self.Y = Y
        self.X_init = X.detach().clone()
        self.Y_init = Y.detach().clone()
        self.Y_mean = Y_mean
        self.Y_std = Y_std

        '''
        Some print statements
        '''

    def set_BO_specs(self,
        objective_func: Optional[object] = None,  
        model: Optional[Model] = None, 
        acq_func: Optional[AcquisitionFunction] = ExpectedImprovement,
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

        self.acq_func = acq_func
        self.minmize = minmize

    def generate_next_point(self, 
        n_candidates: Optional[int] = 1
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
        # Obtain the best value seen so far in Y
        best_value = []
        for i in range(self.n_objectives):
            if self.minmize:
                best_value.append(self.Y[:,i].max())
            else:
                best_value.append(self.Y[:,i].min())
        best_value = torch.tensor(best_value, dtype = dtype)
        
        # Get a new acquicision function and maximize it
        acq_func_current = self.acq_func(self.model, best_f= best_value) 
        unit_bounds = torch.stack([torch.zeros(self.n_dim), torch.ones(self.n_dim)])
        X_new, _ = optimize_acqf(acq_func_current, 
                                bounds= unit_bounds, 
                                q=n_candidates, 
                                num_restarts=10, 
                                raw_samples=100)

        return X_new, acq_func_current

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
            try: 
                Y_new = ut.standardize_X(Y_new_real, self.Y_mean, self.Y_std)
            except KeyError:
                err_msg = "No objective function is specified. The experimental reponse must be provided."
                raise KeyError(err_msg)

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


        
        


    

        





