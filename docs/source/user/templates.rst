===============
Templates
===============

Template for initial DOE
--------------------------

.. code-block:: python

    #%% 1. Import NEXTorch and other packages
    import os, sys
    import numpy as np
    from nextorch import plotting, bo, doe, io, utils

    # set a random seed
    r_seed = 25
    np.random.seed(r_seed)

    #%% 2. Define the design space 
    # the number of dimensions (n_dim) is equal to the number of input parameters
    # Set the input names and units
    X_name_list = ['T', 'Heating rate', 'Time']
    X_units = ['C', 'C/min', 'hr']
    # Add the units
    X_name_with_unit = []
    for i, var in enumerate(X_name_list):
        if not X_units[i]  == '':
            var = var + ' ('+ X_units[i] + ')'
        X_name_with_unit.append(var)

    # Set the output names
    Y_name_with_unit = 'N_Content %'

    # combine X and Y names
    var_names = X_name_with_unit + [Y_name_with_unit]

    # Set the operating range for each parameter
    X_ranges = [[300, 500], 
                [3, 8], 
                [2, 6]] 

    # Set the reponse plotting range
    Y_plot_range = [0, 2.5]

    # Get the information of the design space
    n_dim = len(X_name_list) # the dimension of inputs
    n_objective = 1 # the dimension of outputs


    #%% 3. Define the initial sampling plan
    # Select a design of experimental method first
    # Assume we run Latin hypder cube to create the initial samplinmg
    # Set the initial sampling points, approximately 5*n_dim
    n_init_lhs = 16
    X_init_lhs = doe.latin_hypercube(n_dim=n_dim, n_points=n_init_lhs, seed=r_seed)

    # Convert the sampling plan to a unit scale
    X_init_real = utils.inverse_unitscale_X(X_init_lhs, X_ranges)

    # Visualize the sampling plan,
    # Sampling_3d takes in X in unit scales
    plotting.sampling_3d(X_init_lhs,
                        X_names = X_name_with_unit,
                        X_ranges = X_ranges,
                        design_names = 'LHS')

    print('The predicted new data points in the initial set are:')
    print(io.np_to_dataframe(X_init_real, X_name_with_unit, n = len(X_init_real)))

    # Round up these numbers if needed
    # Now it's time to run those experiments and gather the Y (reponse) values!

Template for active learning iterations
---------------------------------------

.. code-block:: python

    #%% 1. Import NEXTorch and other packages
    import os, sys
    import numpy as np
    from nextorch import plotting, bo, doe, io, utils

    # set a random seed
    r_seed = 25
    np.random.seed(r_seed)

    #%% 2. Define the design space 
    # the number of dimensions (n_dim) is equal to the number of input parameters
    # Set the input names and units
    X_name_list = ['T', 'Heating rate', 'Time']
    X_units = ['C', 'C/min', 'hr']
    # Add the units
    X_name_with_unit = []
    for i, var in enumerate(X_name_list):
        if not X_units[i]  == '':
            var = var + ' ('+ X_units[i] + ')'
        X_name_with_unit.append(var)

    # Set the output names
    Y_name_with_unit = 'N_Content %'

    # combine X and Y names
    var_names = X_name_with_unit + [Y_name_with_unit]

    # Set the operating range for each parameter
    X_ranges = [[300, 500], 
                [3, 8], 
                [2, 6]] 

    # Set the reponse plotting range
    Y_plot_range = [0, 2.5]

    # Get the information of the design space
    n_dim = len(X_name_list) # the dimension of inputs
    n_objective = 1 # the dimension of outputs


    #%% 3. Define the initial sampling plan

    # Import data from an excel file 
    # replace this with your own excel sheet
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NDC_catalyst"))
    file_path = os.path.join(data_path, 'synthesis_data.xlsx')

    # if the excel is under the current folder, uncomment the following code
    # data_path = os.path.abspath(os.getcwd())
    # file_path = os.path.join(data_path, 'synthesis_data.xlsx')

    # Extract the data of interest and also the full data 
    data, data_full = io.read_excel(file_path, var_names = var_names)

    # take a look at the first 5 data points
    print("The input data (first 5): ")
    print(data.head(5))

    print("The full data in the excel file (first 5): ")
    print(data_full.head(5))

    # Split data into X and Y given the Y name
    X_real, Y_real, _, _ = io.split_X_y(data, Y_names = Y_name_with_unit)

    # Extract the iteration index
    trial_no = data_full['Iteration No.']

    # Create the initial sampling plan and responses from the data
    X_init_real = X_real[trial_no==0]
    Y_init_real = Y_real[trial_no==0]

    # Convert the sampling plan to a unit scale
    X_init = utils.unitscale_X(X_init_real, X_ranges)

    # Get the current iteration no
    n_trial_max = np.max(trial_no)


    #%% 4. Initialize an experimental object 
    # Set its name, the files will be saved under the folder with the same name
    Exp = bo.Experiment('NDC_catalyst') 
    # Import the initial data
    Exp.input_data(X_init_real, 
                Y_init_real, 
                X_names = X_name_with_unit, 
                Y_names = Y_name_with_unit, 
                X_ranges = X_ranges, 
                unit_flag = False) #input X and Y in real scales
    # Set the optimization specifications 
    # here we set the objective function, minimization by default
    Exp.set_optim_specs(maximize=True)

    # List for X points in each trial
    X_per_trial = [X_init]

    # Set the sampling points per active learning iteration
    n_points_per_trials_target = 3 

    # Optimization loop
    i_trial = 1 #iteration counter
    while i_trial <= n_trial_max+1:
        
        # Use the n points in the excel if given
        if i_trial <= n_trial_max: 
            n_points_per_trials = np.sum(trial_no == i_trial)
        else:
            n_points_per_trials = n_points_per_trials_target
            
        # Generate the next three experiment point
        X_new_pred, X_new_real_pred, acq_func = Exp.generate_next_point(n_candidates = n_points_per_trials, 
                                                                        acq_func_name = 'qEI')
        
        # Output the predicted points for the next iteration
        print('Starting active learning iteration {}'.format(i_trial))
        print('The predicted new data points for the next iteration are:')
        print(io.np_to_dataframe(X_new_real_pred, X_name_with_unit, n = len(X_new_real_pred)))
        print('Time to run those experiments and gather the Y (reponse) values\n')
        
        # Hit the maximum number of iterations, stop
        if i_trial == n_trial_max+1: 
            break
        
        # Note that the actual data points may be different each time running the script 
        # due to the stochastic nature of the algorithm and rounding errors
        # Now we just extract the experimental data from the last iteration
        X_new_real = X_real[trial_no == i_trial] 
        print('The actual experimental data points in the next iteration are:')
        print(io.np_to_dataframe(X_new_real, X_name_with_unit, n = len(X_new_real)))
        print('Note that the actual experimental points may differ\n')

        # Convert to a unit scale
        X_new = utils.unitscale_X(X_new_real, X_ranges)
        X_per_trial.append(X_new)

        # Run the experiments and get the response data
        Y_new_real = Y_real[trial_no == i_trial]
        print('The actual experimental reponses in the next iteration are:')
        print(io.np_to_dataframe(Y_new_real, Y_name_with_unit, n = len(Y_new_real)))

        # Retrain the model by input the next point into Exp object
        print('Adding these data into training...')
        Exp.run_trial(X_new, X_new_real, Y_new_real)
        print('\n')
        
        i_trial += 1
        
    #%% 5. Plot the reponses versus experiment no. 


