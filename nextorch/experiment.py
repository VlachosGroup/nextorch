"""
nextorch.experiment

Contains Gaussian Processes (GP) and Bayesian Optimization (BO) methods 
"""

import numpy as np
import torch
from torch import Tensor

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

#%% Model training helper functions
def evaluate_objective_func(
    X_unit: MatrixLike2d, 
    X_range: ArrayLike1d, 
    objective_func: object
) -> Tensor:
    """Evaluate the test function

    Parameters
    ----------
    X_unit : MatrixLike2d, matrix or 2d tensor
        X in a unit scale
    X_range : ArrayLike1d, array or 1d tensor
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
    
    
def _predict_pipeline_for_training(model, train_X, Y_mean, Y_std):
    
    '''
    Predict the output value
    return the mean, confidence interval in numpy arrays
    '''
    train_y, train_y_lower,  train_y_upper = ut.predict_surrogate(model, train_X)
    
    train_y_real = ut.inversestandardize_X(train_y, Y_mean, Y_std)
    train_y_real = train_y_real.detach().numpy()
    
    train_y_lower_real = ut.inversestandardize_X(train_y_lower, Y_mean, Y_std)
    train_y_lower_real = train_y_lower_real.detach().numpy()
    
    train_y_upper_real = ut.inversestandardize_X(train_y_upper, Y_mean, Y_std)
    train_y_upper_real = train_y_upper_real.detach().numpy()
    
    return train_y_real, train_y_lower_real, train_y_upper_real
        
def predict_surrogate(model, test_x):
    '''
    Input a Gaussian process model 
    return the mean, lower confidence interval and upper confidence intervale 
    from the postierior
    '''
    
    test_x = copy.deepcopy(test_x)
    test_x = torch.tensor(test_x, dtype = dtype)
    
    posterior = model.posterior(test_x)
    
    test_y_mean = posterior.mean
    test_y_lower, test_y_upper = posterior.mvn.confidence_region()
    
    return test_y_mean, test_y_lower, test_y_upper

def _predict_pipeline_for_testing(model, test_X,  mesh_size, Y_mean, Y_std):
    
    '''
    Predict 2d mesh values from surrogates 
    Generate plots if required
    Return outputs Y in 2d numpy matrix 
    '''
    # predict the mean for ff model, returns a standardized 1d tensor
    test_y, test_y_lower, test_y_upper = ut.predict_surrogate(model, test_X)
    # Inverse the standardization and convert 1d y into a 2d array
    test_Y_real = ut.transform_plot2D_Y(test_y, Y_mean, Y_std, mesh_size)
    
    return test_Y_real

    


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
            Y_mean = Y.mean(axis = 0)
            Y_std = Y.std(axis = 0)

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
        Y_new_real: Optional[Tensor] = None,
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

    

        





