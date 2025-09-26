import numpy as np
import matplotlib as mpl
from load import *
from outliers import *

if __name__ == "__main__":

    #Exercicio 2

    '''
    part_data = load_part_data(0)
    print("\nParticipant 0 data:\n")
    print(part_data)
    '''

    #Exercicio 3.1

    variable = "Acceleration"

    data = load_data()
    var_trans_data = variable_module(data, variable)
    boxplot_variable(data, var_trans_data, variable)

    #Exercicio 3.2

    outlier_density(data, var_trans_data)

    #Exercicio 3.3

    activity = 12  # Stand -> Walk
    outlier_idxs = z_score(data, var_trans_data, activity, 3)
    plot_zscore_outliers(data, var_trans_data, activity, outlier_idxs)
