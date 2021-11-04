===============
Templates 
===============


The templates are mainly for lab experiments and human-in-the-loop optimizatin. 

Step-by-step Guide
---------------------

1. Use `template_initial_doe.py` to generate points for the initial experimental design
- The goal is to sample the design space efficiently with an initial DOE
- Determine the number of parameters (n_dim) for your system. Change the variable names, units and ranges in section 2
- Change the DOE method if needed. We use Latine Hypercube (LHS) by default where 5*n_dim points are needed approximately


2. Use `template_active_learning_iterations.py` for the following active learning template_active_learning_iterations
- The goal is to using active learning to find optimum in the design space with fewer sampling points
- Change the section 2 to be identifical as the section 2 in `template_initial_doe.py`
- Save the experimental data in an excel or csv file containing the columns named `Iteration No.` and `Experimental No.`. The parameters and reponses should also be included. The column names should be consistent with the values in `X_name_with_unit` and `Y_name_with_unit` (in section 2)
- Supply the name of excel or csv file in section 3
- Set `n_points_per_trials_target` for the next experiment iteration in section 4
- Perform the experiments based on the perdicted points
- Update the excel or csv file with the new data and reponses value
- Check the current optimum and iterate until convergence 
