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
    X_real = inverse_unitscale_X(X_unit_np, X_range_np)
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
    """Experiment object, consists of a set of trial points"""

    def ___init___(
        self,
        X: Tensor,
        Y: Tensor,
        X_ranges: Tensor,
        name: Optional[str] = 'Experiment', 
        X_names: Optional[List[str]] = None,
        Y_names: Optional[List[str]] = None, 
        preprocessed: Optional[bool] = False
    ):

        self.X = X
        self.Y = Y
        self.X_init = X.detach().clone()
        self.Y_init = Y.detach().clone()


        self.n_dim = X.shape[1] 
        self.n_points = X.shape[0]
        self.n_objectives = Y.shape[1]

        self.name = name
        
        if X_names is None:
            X_names = ['X' + str(i+1) for i in range(self.n_dim)]

        if Y_names is None:
            Y_names = ['Y' + str(i+1) for i in range(self.n_objectives)]

        self.X_names = X_names
        self.Y_names = Y_names

        self.Y_means = torch.zeros(self.n_objectives)
        self.Y_std = torch.ones(self.n_objectives)

        '''
        Some print statements
        '''


    def preprocess(self):
        pass




    def set_BO_specs(
        self,
        objective_func: object,  
        model: Optional[Model] = None, 
        acq_func: Optional[AcquisitionFunction] = ExpectedImprovement,
        minmize: Optional[bool] = True,
    ):  
        self.objective_func = objective_func

        if model is None:
            self.model = create_and_fit_gp(self.X, self.Y)

        self.acq_func = acq_func
        self.minmize = minmize

    def generate_next_point(self) -> Tuple[Tensor, Tensor]:

        best_value = []
        for i in range(self.n_objectives):
            if self.minmize:
                best_value.append(self.Y[:,i].max())
            else:
                best_value.append(self.Y[:,i].min())
        best_value = torch.tensor(best_value, dtype = dtype)

        unit_bounds = torch.stack([torch.zeros(self.n_dim), torch.ones(self.n_dim)])

        acq_func_val_current = self.acq_func(self.model, best_f= best_value) 
        X_new, acq_value = optimize_acqf(acq_func_val_current, bounds= unit_bounds, 
                                        q=1, num_restarts=10, raw_samples=100)

        return X_new, acq_func_val_current

    def run_trial(self):

        # One Iteration
        X_new, acq_func_val_current = self.generate_next_point()
        # new_X, new_Y are 2D tensors
        Y_new_real = ut.evaluate_objective_func(X_new, self.X_range, self.objective_func)

        Y_new = ut.standardize_X(Y_new_real, self.Y_mean, self.Y_std)
                
        # Cat all the training data
        self.X = torch.cat((self.X, X_new))
        self.Y = torch.cat((self.Y, Y_new))

        self.n_points += 1
        
        # Add the new point into the model
        self.model = fit_with_new_observations(self.model, X_new, Y_new)

        
        return self.X, self.Y



        





