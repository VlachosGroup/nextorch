'''
# MATLAB Code written by: Pierre Desir
# Adapted in Python by: Yifan Wang
# Date: 01-16-2019

# This code takes in 3 inputs
 
# T - Rxn temperature [°C]
# pH - Rxn pH 
# tf - final residence time (min)
# Example: FRU_PFR_Model_function(100, 0.7, 1000)
 
#This MatLab Code solves the PFR model for the acid-catalyzed dehydration of
#fructose to HMF using HCl as the catalyst and generates the kinetic data
#for conversion, yield, and selectivity of the species as a function of
#temperature, pH, and time. All the kinetic and thermodynamic parameters 
#were taken from the kinetic model by T. Dallas Swift et al. ACS Catal 
#2014, 4,259-267.
'''

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint, ode

#%%-------------------------------------------------------------------------
# Define the variables and units
Variable_list = ['T_degC', 'pH', 'tf']
Units = ['C', '', 'min']
var_with_unit = []
for i, var in enumerate(Variable_list):
    if not Units[i]  == '':
        var = var + ' ('+ Units[i] + ')'
    var_with_unit.append(var)

#Reactor functions
def Reactor(**conditions):
    
    '''
    Reactor model under certain temperature, pH and residence time tf
    Returns the final HMF yield
    '''
    T_degC = conditions['T_degC (C)'] # reaction temperature in C
    pH = conditions['pH'] # reaction pH
    tf = conditions['tf (min)'] #Final time point [min]
    t0 = 0 #Initial time point [min]
    Fru0 = 1 #Normalized initial fructose concentration (always equal to 1)

    def PFR(t,C, T_degC, pH):
        '''
        #The "PFR" function describes the species conservative equation as a
        #set of ODEs at T
        '''    
        #CONSTANTS
        R = 8.314 #Ideal gas law constant [J/mol-K]
        Tm = 381 #Mean temperature of all Rxn [K]
        

        #OTHER MODEL PARAMETER
        T = T_degC + 273 # reaction temperarure in K
        C_Hplus = 10**(-pH) #H+ concentraction [mol/L]
        C_H2O = 47.522423477254065 + 0.06931572301966918*T - 0.00014440077466393135* (T**2) #Water 
        
        #concentration as a function of temperature from 25 °C to 300 °C
        #[mol/L]
           
        
        #TAUTOMER PARAMETERS
        #Enthalpy of Rxn b/t open-chain fructose and tautomers
        #Note: a = alpha; b = beta; p = pyrannose; f = furannose
        delH_bp = -30.2e3 #[J/mol]
        delH_bf = -19e3 #[J/mol]
        delH_ap = -5.5e3 #[J/mol]
        delH_af = -14.2e3 #[J/mol]
        
        #Equilibrium constants b/t open-chain fructose and tautomers at 303 K
        K_bp303 = 59.2
        K_bf303 = 26.4
        K_ap303 = 0.6
        K_af303 = 6.4
        
        #Equilibirium constants b/t open-chain fructose and tautomers as a 
        #function of temperature
        K_bp = K_bp303*np.exp(-(delH_bp/R)*(1/T-1/303))
        K_bf = K_bf303*np.exp(-(delH_bf/R)*(1/T-1/303))
        K_ap = K_ap303*np.exp(-(delH_ap/R)*(1/T-1/303))
        K_af = K_af303*np.exp(-(delH_af/R)*(1/T-1/303))
        
        #Furanose fraction at equilibirum as a function of temperature
        phi_f = (K_af + K_bf)/(1 + K_af + K_bf + K_ap + K_bp)
        
        # ACTIVATIONS ENERGIES FOR RXN1,RXN2,...,RXN5
        Ea = np.array([127, 133, 97, 64, 129]) * (10**3) #[J/mol]
        
        #NTURAL LOG OF RXN RATE CONSTANTS AT 381 K FOR RXN1,RXN2,...,RXN5
        lnk381 = np.array([1.44, -4.22, -3.25, -5.14, -4.92])
         
        #RXN RATE CONSTANTS FOR RXN1,RXN2,...,RXN5 AS A FUNCTION OF 
        #TEMPERATURE
        k = np.zeros(5)
        for i in range(len(k)):
            k[i] = np.exp(lnk381[i]-(Ea[i]/R)*(1/T-1/Tm)) #[min^-1.M^-1]
        
        #RXN RATES FOR THE RXN NETWORK OF FRUCTOSE DEHYDRATION
        #Note: C[0] = Normalized Fructose concentration; C[1] = Normalized
        #HMF concentration; C[2] = Normalized LA concentration; 
        #C[3] = Normalized FA concentration;
        Rxn = np.zeros(5) #[mol/L-min]
        Rxn[0] = k[0]*phi_f*C[0]*C_Hplus/C_H2O #[mol/L-min]
        Rxn[1] = k[1]*C[0]*C_Hplus #[min^-1]
        Rxn[2] = k[2]*C[1]*C_Hplus #[min^-1]
        Rxn[3] = k[3]*C[1]*C_Hplus #[min^-1]
        Rxn[4] = k[4]*C[0]*C_Hplus #[min^-1]
         
        #SPECIES CONSERVATIVE EQUATIONS
        #Notation: rhs = dC/dt
        rhs = np.zeros(4) #[mol/L-min]
        rhs[0] = (-Rxn[0]-Rxn[1]-Rxn[4]) #Fructose
        rhs[1] = (Rxn[0]-Rxn[2]-Rxn[3]) #HMF
        rhs[2] = Rxn[2] #LA
        rhs[3] = (Rxn[2]+Rxn[4]) #FA
        
        return rhs
      
    #%% SOLVING FOR THE PFR MODEL at certain temperature T_K
    C0 = np.array([Fru0, 0, 0, 0])
    
    # Construct the ode solver, ode45 with varying step size
    sol = []
    def solout(t, y):
        sol.append([t, *y])
    solver = ode(PFR).set_integrator('dopri5', rtol  = 1e-6, method='bdf')
    solver.set_solout(solout)
    #feed in argumenrs and initial conditions for odes
    solver.set_initial_value(C0, t0).set_f_params(*(T_degC, pH)) 
    solver.integrate(tf)
    sol = np.array(sol)
    
    Tau = sol[:,0] #time steps
    Conc = sol[:,1:] #concentration matrix
    
    #Only take the final results and no round up 
    Fru = Conc[:,0]
    HMF = Conc[:,1]
    HMF_Yield = 100*HMF
    HMF_Yield_final = HMF_Yield[-1]
    Conv_final = 100*(1-Fru[-1])
    HMF_Select_final = 100*HMF_Yield_final/Conv_final
    
    #RESULTS
#    Fru = np.around(Conc[:,0], decimals = 4) #Fructose normalized concentration 
#    HMF = np.around(Conc[:,1], decimals = 4) #HMF normalized concentration 
#    LA = np.around(Conc[:,2], decimals = 4) #Levulinic acid (LA) concentration 
#    FA = np.around(Conc[:,3], decimals = 4) #Formic acid (FA) concentration 
#    Conv = np.around(100*(1-Fru), decimals = 4) #Fructose conversion [%]
#    HMF_Yield = 100*HMF #HMF yield [%]
#    HMF_Select = 100*HMF_Yield/Conv #HMF selectivity [%]
#    LA_Yield = 100*LA #LA yield [%]
#    FA_Yield = 100*FA #FA yield [%]
#    HMF_Yield_final = HMF_Yield[-1]
#    Conv_final = Conv[-1]
#    HMF_Select_final = HMF_Select[-1]
     
    # #OPTIMAL CONDITIONS FOR MAX HMF YIELD
    # Max_HMF_Yield = max(HMF_Yield) #Maximum HMF Yield [%]
    # index_range = np.where(HMF_Yield == Max_HMF_Yield)[0] # Index of matrix 
    # #element where the HMF yield is at its max value
    # index = index_range[max(np.where(Conv[index_range] == max(Conv[index_range]))[0])]
    # #index of matrix element for optimal conditions for max HMF yield
    # Tau_opt = Tau[index] #Optimal residence time to reach maximum HMF 
    # #yield [min]
    # Opt_Conv = np.around(Conv[index], decimals = 0) #Fructose conversion at max HMF yield [%]
    # Opt_Select = HMF_Select[index] #HMF selectivity at max HMF yield [%]
    
    # #REPORTING OPTIMAL CONDITIONS
    # Opt_Cond = (T_degC, Tau_opt, Max_HMF_Yield, Opt_Conv, Opt_Select)
    # #Temperature, optimal residence time, max HMF yield, conversion at max
    # #HMF yield, HMF selectivity at max HMF yield

    return HMF_Yield_final, HMF_Select_final
