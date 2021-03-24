===================
Experiment
===================

.. currentmodule:: nextorch.bo

In NEXTorch, We use :code:`Experiment` objects to store the data and the setting of the optimization loop. We integrate 
most acquisition functions and GP models from the upstream BoTorch with :code:`Experiment` objects. Experiment objects 
are the core of NEXTorch where data are preprocessed, GP models are trained, used for prediction, and the optima are located. 
There are several variants of the Experiment class depending on the application. At the higher level, users interact with the 
:code:`Experiment` class, :code:`WeightedMOOExperiment` class, and :code:`EHVIMOOExperiment` for sequential single-objective 
optimization, MOO with the weighted sum method, and MOO with EHVI, respectively. Besides, two additional classes, :code:`COMSOLExperiment` 
and :code:`COMSOLMOOExperiment` are provided to perform automatic optimization by integrating the NEXTorch with COMSOL Multiphysics\ |reg|.

Example
------------

:code:`Experiment` class
^^^^^^^^^^^^^^^^^^^^^^^^
Initialize a :code:`Experiment` object with :code:`input_data` to specify the inputs of parameters and response, 
the range and name of them, and their conditions.

.. code-block:: python

    from nextorch import bo

    X_names = ["param_1", "param_2", "param_3"]
    X_ranges = [[1.0, 25.0], [1.0, 25.0], [1.0, 25.0]]
    X_units = ["mm", "mm", "mm"]
    X_real = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [25.0, 25.0, 25.0]])
    Y_real = np.array([[0.5], [0.2], [0.7]])

    exp = bo.Experiment("Experiment")
    exp.input_data(X_real=X_real, Y_real=Y_real)

Additionally, we can specify more conditions:

..code-block:: python

    exp.input_data(X_real=X_real, Y_real=Y_real, X_ranges=X_ranges, X_names=X_names, standardized=True)

.. note::

    :code:`X_real` and :code:`Y_real` are the required inputs for :code:`Experiment` object.
    
Set the optimization conditions using :code:`set_optim_specs`

.. code-block:: python

    def simple_1d(X):
        """
        1D function y = (6x-2)^2 * sin(12x-4)
        """

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

    objective_func = simple_1d

    exp.set_optim_specs(objective_func=objective_func)

.. note::

    :code:`set_optim_specs` by default maximizing the objective function. To minimize, set :code:`maximize` to :code:`False`.

    .. code-block:: python

        exp.set_optim_specs(objective_func=objective_func, maximize=False)

:code:`WeightedMOOExperiment` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In :code:`WeightedMOOExperiment`, an additional :code:`weights` for different objectives needs to be specify in :code:`set_optim_specs`.

.. code-block:: python

    weights_obj = np.linspace(0, 1, 21)

    exp_weighted = bo.WeightedMOOExperiment("Weighted Experiment")
    exp_weighted.input_data(X_real=X_real, Y_real=Y_real, X_ranges=X_ranges, X_names=X_names)
    exp_weighted.set_optim_specs(objective_func=objective_func, maximize=True, weights=weights_obj)

:code:`EHVIMOOExperiment` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On the other hand, we can use an :code:`EHVIMOOExperiment` object to perform optimization using expected hypervolume 
improvement as aquisition funciton. It requires all key components as :code:`Experiment`. Additionally, :code:`ref_point` 
is required for :code:`set_ref_point` function.

.. code-block:: python

    ref_point = [10.0, 10.0]

    exp_ehvi = bo.EHVIMOOExperiment("MOO Experiment")
    exp_ehvi.input_data(X_real=X_real, Y_real=Y_real, X_ranges=X_ranges, X_names=X_names)
    exp_ehvi.set_ref_point(ref_point)
    exp_ehvi.set_optim_specs(objective_func=objective_func, maximize=True)

.. note::

    :code:`ref_point` defines a list of values that are slightly worse than the lower bound of objective values, where 
    the lower bound is the minimum acceptable value of interest for each objective. It would be helpful if the user 
    know the rough values using domain knowledge prior to optimization. 

:code:`COMSOLExperiment` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :code:`COMSOLExperiment` and :code:`COMSOLMOOExperiment` are designed for the integration with COMSOL Multiphysics\ |reg|.
It requires all key components as :code:`Experiment`. Additionally, the :code:`X_names` and :code:`X_units` are required. 
Instead of specifying objective function, the optimized file name,the installed location of COMSOL Multiphysics\ |reg|,  
the output location of the simulation, and the selected column in the output file are needed.

.. code-block:: python

    file_name = "comsol_example" 
    comsol_location =  "~/comsol54/multiphysics/bin/comsol" 
    output_file = "simulation_result.csv"

    exp_comsol = bo.COMSOLExperiment("COMSOL Experiment")
    exp_comsol.input_data(X_real=X_real, Y_real=Y_real, X_ranges=X_ranges, X_names=X_names, X_units=X_units)
    exp_comsol.set_optim_specs(file_name, comsol_location, output_file, comsol_output_col=2, maximize=True)

.. note::

    To use :code:`COMSOLExperiment` and :code:`COMSOLMOOExperiment` class, you need to have or purchase your own valid 
    license for COMSOL Multiphysics\ |reg|.

For all of discussed classes, :code:`run_trials_auto` can automatically perform the optimization.

.. code-block:: python

    n_trials=30

    exp.run_trials_auto(n_trials=n_trials)
    exp_weighted.run_trials_auto(n_trials=n_trials, acq_func_name='EI')
    exp_ehvi.run_trials_auto(n_trials=n_trials)
    exp_comsol.run_trials_auto(n_trials=n_trials)

.. note::

    The acquisition function used can be specified for :code:`run_trials_auto` using :code:`acq_func_name`.

Get the final optimal using :code:`get_optim` for all classes

.. code-block:: python

    y_opt, X_opt, index_opt = exp.get_optim()

For more details, see :code:`nextorch.bo`.

.. autosummary::
    :nosignatures:

    Experiment
    WeightedMOOExperiment
    EHVIMOOExperiment
    COMSOLExperiment
    COMSOLMOOExperiment

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN