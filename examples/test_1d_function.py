# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 17:02:03 2019

@author: Yifan Wang


Example of Bayesian Optimization for 1D example
Use Gaussian Process regression to fit the model 
Use one type of acqucision functions to predict the next infill point
For minmization
"""
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_torch

import matplotlib.pyplot as plt 
import matplotlib

font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2


# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dtype = torch.float

def test_function(X):
    '''
    1D function f(x) = (6x-2)^2 * sin(12x-4)
    X can be np array or a list
    y is a np array
    '''
    try:
        X.shape[1]
    except:
        X = np.array(X)

    if len(X.shape)<2:
        
        X = np.array([X])
        
    y = np.array([],dtype=float)
    
    for i in range(X.shape[0]):
        
        ynew = (X[i]*6-2)**2*np.sin((X[i]*6-2)*2) 
        y = np.append(y, ynew)
    
    y = y.reshape(X.shape)
    
    return y



#%% Set up inputs, bounds and dimension   

#Setup Initial Dataset
dim = 1

train_X = np.array([0, 0.25, 0.5, 0.75])
train_X = np.array([train_X]).T
Y = test_function(train_X)


# Convert to torch tensors
train_X = torch.tensor(train_X, dtype=dtype)
Y = torch.tensor(Y, dtype=dtype)

# Normalize Y
Y_mean = Y.mean()
Y_std = Y.std()
train_Y = (Y - Y_mean) / Y_std

all_train_X = train_X.clone()
all_train_Y = train_Y.clone()
bounds = torch.tensor([[0.], [1.0]], dtype=dtype)

# test model on 101 regular spaced points on the interval [0, 1]
test_X = np.linspace(0, 1, 1000)
test_Y = test_function(test_X)[0]
test_X = torch.tensor(test_X, dtype = dtype)

test_Y_original = torch.tensor(test_Y, dtype = dtype)
test_Y = (test_Y_original - Y_mean) / Y_std

#%% Plot functions
def plot_testing(model, test_X, train_X, train_Y,  test_Y = None, new_X = None, new_Y = None):
    '''
    Test the surrogate model with model, test_X and new_X
    '''

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))

    with torch.no_grad():
        # compute posterior
        posterior = model.posterior(test_X)
        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior.mvn.confidence_region()
        
        # Plot the groud truth test_Y if provided
        ax.plot(test_X.cpu().numpy(), test_Y.cpu().numpy(), 'k--', label = 'Objective f(x)')
        # Plot posterior means as blue line
        ax.plot(test_X.cpu().numpy(), posterior.mean.cpu().numpy(), 'b', label = 'Posterior Mean')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_X.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5, label = 'Confidence')
        
        # Plot training points as black stars
        ax.scatter(train_X.cpu().numpy(), train_Y.cpu().numpy(), s =120, c= 'k', marker = '*', label = 'Initial Data')
         # Plot the new infill points as red stars
        if not type(new_X) == type(None):    
            ax.scatter(new_X.cpu().numpy(), new_Y.cpu().numpy(), s = 120, c = 'r', marker = '*', label = 'Infill Data')
        
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.show()

def plot_acq_func(acq_func, test_X, train_X, new_X = None):
    # compute acquicision function values at test_X
    test_acq_val = acq_func(test_X.view((test_X.shape[0],1,dim)))
    train_acq_val = acq_func(train_X.view((train_X.shape[0],1,dim)))

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
    with torch.no_grad():
        ax.plot(test_X.cpu().numpy(), test_acq_val.detach(), 'b-', label = 'Acquistion (EI)')
        # Plot training points as black stars
        ax.scatter(train_X.cpu().numpy(), train_acq_val.detach(), s = 120, c= 'k', marker = '*', label = 'Initial Data')
         # Plot the new infill points as red stars
        if not type(new_X) == type(None):
            new_acq_val = acq_func(new_X.view((new_X.shape[0],1,dim)))
            ax.scatter(new_X.cpu().numpy(), new_acq_val.detach(),  s = 120, c ='r', marker = '*', label = 'Infill Data')
    
    ax.ticklabel_format(style = 'sci', axis = 'y', scilimits = (-2,2) )
    ax.set_xlabel('x')
    ax.set_ylabel(r'$ \alpha$')    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.show()
    
def _get_and_fit_simple_gp(Xs, Ys, **kwargs):
    
    model = SingleTaskGP(train_X=Xs, train_Y=Ys)
    model.train();
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(all_train_X)
    mll.train();
    fit_gpytorch_torch(mll);
    mll.eval();
    model.eval();
    return model    

def _fit_with_new_observations(model, Xs, Ys, **kwargs):
    
    # Add the new point into the model
    model = model.condition_on_observations(X=Xs, Y=Ys)
    model.train();
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(all_train_X)
    mll.train();
    fit_gpytorch_torch(mll);
    mll.eval();
    model.eval();
    
    return model
    
def _optimize_single_loop(model, test_X, all_train_X, all_train_Y, knowledge_base = True):
    
    # One Iteration
    # Generate a newfill point based on the acquisition functions
    # Train the acquisition function
    if knowledge_base == True:
        best_value = model.posterior(test_X).mean.min()
    else: best_value = all_train_Y.min()
    
    #acq_func = UpperConfidenceBound(model, beta = 0.1)
    acq_func = ExpectedImprovement(model, best_f= best_value, maximize = False) #all_train_Y.max())
    new_X, acq_value = optimize_acqf(acq_func, bounds=bounds, 
                                         q=1, num_restarts=10, raw_samples=100)
    new_Y = (torch.tensor(test_function(new_X), dtype = dtype) - Y_mean) / Y_std
    
    plot_acq_func(acq_func, test_X, all_train_X, new_X)
    
    # Cat all the training data
    all_train_X = torch.cat((all_train_X, new_X))
    all_train_Y = torch.cat((all_train_Y, new_Y))
    
    # Add the new point into the model
    model = _fit_with_new_observations(model, new_X, new_Y)
    
    # Plot the model
    plot_testing(model, test_X, all_train_X, all_train_Y, test_Y = test_Y, new_X = new_X, new_Y = new_Y)
    
    return model, all_train_X, all_train_Y


    
#%% Main Optimization loop 

# Initialization    
# Fit a Gaussian Process model
model = _get_and_fit_simple_gp(all_train_X, all_train_Y)

# Plot the model
plot_testing(model, test_X, all_train_X, all_train_Y, test_Y = test_Y)

# Optimization loop
n = 10
for i in range(n):
    model, all_train_X, all_train_Y = _optimize_single_loop(model, test_X, all_train_X, all_train_Y)
      
