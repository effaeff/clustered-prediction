import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import misc

misc.to_local_dir(__file__)




# ### File-Infos und Parameter, für jede Messung anpassen

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_1001.txt', 13499.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.35
# dphi_y = 2.0
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.8 #1.18
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_2001.txt', 13900.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 2.35
# dphi_y = 2.8
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_3001.txt', 16393.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.35
# dphi_y = 1.65
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.0 #1.18
# runout_len = 1.5 # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_4002.txt', 15153.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.25
# dphi_y = 1.85
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_5001.txt', 12709.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.0
# dphi_y = 0.35
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 7.5 #1.18
# runout_len = 1.5 # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_6001.txt', 13056.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.4
# dphi_y = 0.
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2. #1.18
# runout_len = 1.5 # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_7002.txt', 12396.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.5
# dphi_y = 0.75
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2. #1.18
# runout_len = 1.5 # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_8002.txt', 16359.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.5
# dphi_y = 0.75
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2. #1.18
# runout_len = 1.5 # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_9001.txt', 15612.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.20
# dphi_y = 2.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2. #1.18
# runout_len = .5 # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_10001.txt', 13761.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.5
# dphi_y = 0.75
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 5. #1.18
# runout_len = 1.5 # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_11001.txt', 14959.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.20
# dphi_y = 2.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5 #1.18
# runout_len = .5 # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_12002.txt', 12598.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.75
# dphi_y = 1.0
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5 #1.18
# runout_len = .25 # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_13001.txt', 15323.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.7
# dphi_y = 2.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 3.5 #1.18
# runout_len = 1. # seconds

###################################################################


# filename_exp, spsp = './_data/Cluster_Sim_V0_14001.txt', 12025.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.60
# dphi_y = 2.65
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 3.5 #1.18
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_15001.txt', 16870.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.35
# dphi_y = 0.9
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2. #1.18
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_16001.txt', 16818.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.9
# dphi_y = 2.
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5 #1.18
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_17001.txt', 13171.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.5
# dphi_y = 2.0
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.8
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_18001.txt', 14005.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.3
# dphi_y = 0.15
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 4.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_19001.txt', 13840.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.5
# dphi_y = -0.35
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_20002.txt', 13620.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 2.2
# dphi_y = 2.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_21001.txt', 16958.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 2.3
# dphi_y = 2.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_22002.txt', 12239.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 3.
# dphi_y = 3.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 5.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_23002.txt', 12903.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 3.
# dphi_y = 3.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.75
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_24001.txt', 12700.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 2.
# dphi_y = 2.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_25001.txt', 14201.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.5
# dphi_y = 1.9
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 4.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_26001.txt', 16116.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.2
# dphi_y = 0.7
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 4.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_27001.txt', 13656.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.4
# dphi_y = 0.7
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 4.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_28001.txt', 14854.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.6
# dphi_y = -0.4
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_29001.txt', 16695.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.6
# dphi_y = -0.4
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_30001.txt', 16254.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.4
# dphi_y = -0.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_31001.txt', 16043.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.8
# dphi_y = 1.6
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_32001.txt', 15387.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.
# dphi_y = 1.3
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.25
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_33001.txt', 15724.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.5
# dphi_y = -0.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_34001.txt', 14406.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.2
# dphi_y = 1.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_35001.txt', 15041.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.
# dphi_y = 1.
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_36001.txt', 15065.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.8
# dphi_y = 0.9
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_37001.txt', 12812.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.6
# dphi_y = 0.9
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_38001.txt', 15528.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.6
# dphi_y = -0.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_39001.txt', 15828.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.4
# dphi_y = 0.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.25
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_40001.txt', 12541.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.3
# dphi_y = -0.3
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.25
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_41001.txt', 13214.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.1
# dphi_y = 0.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_42001.txt', 13003.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.2
# dphi_y = 1.4
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_43001.txt', 16650.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -1.
# dphi_y = -0.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 3.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_44001.txt', 14314.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.3
# dphi_y = 0.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_45001.txt', 14663.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.2
# dphi_y = -0.1
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 4.
# runout_len = 1. # seconds

###################################################################

# # missing
# filename_exp, spsp = './_data/Cluster_Sim_V0_46001.txt', 15887.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.2
# dphi_y = -0.1
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 4.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_47001.txt', 15447.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.8
# dphi_y = 2.1
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_48001.txt', 12377.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.5
# dphi_y = 0.9
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_49001.txt', 16202.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.5
# dphi_y = -0.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_50001.txt', 14790.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.2
# dphi_y = -0.1
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_51001.txt', 15206.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.2
# dphi_y = -0.1
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 4.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_53001.txt', 14521.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.7
# dphi_y = 0.7
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_54001.txt', 13356.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.5
# dphi_y = 0.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_55001.txt', 13413.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = -0.2
# dphi_y = -0.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_56001.txt', 13323.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.7
# dphi_y = 0.7
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_57001.txt', 12121.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.8
# dphi_y = 1.9
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_58001.txt', 14602.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.8
# dphi_y = 1.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_59001.txt', 12187.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 2.0
# dphi_y = 2.6
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_60002.txt', 16488.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.9
# dphi_y = 1.7
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.5
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_61001.txt', 14156.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 1.9
# dphi_y = 1.6
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 1.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_62003.txt', 14072.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.6
# dphi_y = 0.8
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_63001.txt', 15978.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 2.2
# dphi_y = 2.2
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################

# filename_exp, spsp = './_data/Cluster_Sim_V0_64001.txt', 14461.0
# amp_x = 0.00002
# amp_y = 0.00002
# dphi_x = 0.6
# dphi_y = 0.5
# initial_parameters_x = [amp_x, 1*spsp / 60.0, dphi_x, amp_x, 2*spsp / 60.0, dphi_x, amp_x, 4*spsp / 60.0, dphi_x, amp_x, 6*spsp / 60.0, dphi_x]
# initial_parameters_y = [amp_y, 1*spsp / 60.0, dphi_y, amp_y, 2*spsp / 60.0, dphi_y, amp_y, 4*spsp / 60.0, dphi_y, amp_y, 6*spsp / 60.0, dphi_y]
# runout_start_left  = 2.
# runout_len = 1. # seconds

###################################################################



export_filename = filename_exp.split('/')[-1][:-4] + str('_')
print(f'Looking at {export_filename[:-1]}')

# Read experiment

save_npz = False

if os.path.isfile('./_data/' + export_filename[:-1] + '.npz'):
    data = np.load('./_data/' + export_filename[:-1] + '.npz',allow_pickle=True)["data"]
    save_npz = True
else:
    with open(filename_exp, 'r') as file:
        lines = file.readlines()
        data = np.array([[float(x) for x in line.replace(',','.').split('\t')] for line in lines[8:]])
        np.savez(f'./_data/{export_filename[:-1]}.npz', data=data[:,[0,5,6]])


dt = data[1,0] - data[0,0]
sampling_rate = 1.0 / dt

runout_len = int(runout_len / dt *.925)

if save_npz == False:
    data = data[:,[0,5,6]]
else:
    data = data[:,[0,1,2]]
# Anfang des Zeitkanals nullen
data[:,0] -= data[0,0]
mean_idx = 30000

# Statischen Anteil in Auslenkungsmessung in x und y abziehen (Mittelwert wird subtrahiert)
data[:,1] -= data[:mean_idx,1].mean(axis=0)
data[:,2] -= data[:mean_idx,2].mean(axis=0)
#### Rumspielen!, falls offset in zur Auswertung genutzter Daten; ggfs. auskommentieren
data[:,2] += np.median(data[:mean_idx,2])


# Find process (Autom. Bestimmung des relevanten Bereich kurz vor und nach dem Eingriff)
threshold = np.max(data[:,2]) / 2.
threshold_indices = np.where(data[:,2] > threshold)[0]
process_start = np.min(threshold_indices) 
process_end   = np.max(threshold_indices) 
process_index = (process_start + process_end) // 2

# Filter runout (Gibt den Bereich vor dem Eingriff an, anhand dem die optimierung der Rundlauffehler-Approximationsfunktion erfogt. diese ist kurz vor de Eingriff, um Fehler infolge von Drehzahlabweichungen zu minimieren)
runout_start_left  = int(runout_start_left/dt)
data_runout = data[runout_start_left:runout_start_left+runout_len]



# Approximationsfunktion, deren Parameter durch num. Optimierung parametriert werden (alle X-Werte)
def calc_runout(x, time_data):
    runout = x[0] * np.sin(2.0 * np.pi * x[1] * time_data + x[2])
    runout += x[3] * np.sin(2.0 * np.pi * x[4] * time_data + x[5])
    runout += x[6] * np.sin(2.0 * np.pi * x[7] * time_data + x[8])
    runout += x[9] * np.sin(2.0 * np.pi * x[10] * time_data + x[11])
    return runout

# Gibt den Fehler zw. Messung und Approximation zurück, ruft Approximationsfunktion auf 
def min_func(x, data_x, data_y):
    return np.sum((data_y - calc_runout(x, data_x))**2)

# Aufruf des Optimierers mittels "minimize", ruft min_func auf und übergibt Parameter
result_x = minimize(min_func, initial_parameters_x, (data_runout[:,0], data_runout[:,1]))
result_y = minimize(min_func, initial_parameters_y, (data_runout[:,0], data_runout[:,2]))
runout = np.column_stack((calc_runout(result_x.x, data[:,0]), calc_runout(result_y.x, data[:,0])))
data_filtered = data.copy()
data_filtered[:,1] -= runout[:,0]
data_filtered[:,2] -= runout[:,1]



###################### save filtered displacement-measurements


np.savez(f'./_eval/{export_filename}filtered.npz', time=data_filtered[:,0], dx=data_filtered[:,1], dy=data_filtered[:,2])

###################### plots

# Plot experiment
plt.figure()
plt.title('Rohdaten')
plt.plot(data[:,0], data[:,1], label='dx_exp_raw')
plt.plot(data[:,0], data[:,2], label='dy_exp_raw')
plt.axvspan(data[runout_start_left,0], data[runout_start_left+runout_len,0], color='r', alpha=0.5, label='Area runout Best.')
plt.legend()


# Plot runout dx
plt.figure()
plt.title('runout dx (Exp. + Approx.)')
plt.plot(data[:,0], data[:,1], label='dx_exp')
plt.plot(data[:,0], runout[:,0], label='runout dx')
plt.axvspan(data[runout_start_left,0], data[runout_start_left+runout_len,0], color='r', alpha=0.5)
plt.legend()


# Plot runout dy
plt.figure()
plt.title('runout dy (Exp. + Approx.)')
plt.plot(data[:,0], data[:,2], label='dy_exp_raw')
plt.plot(data[:,0], runout[:,1], label='runout dy')
plt.axvspan(data[runout_start_left,0], data[runout_start_left+runout_len,0], color='r', alpha=0.5)
plt.legend()


####### Vgl. Rundlauf roh und gefiltert (vor Eingriff)

#Plot process window: raw and filtered dx
plt.figure()
plt.title('Roh + gefilterte dx Daten')
plt.plot(data[:,0], data[:,1], label='dx_exp_raw')
plt.plot(data_filtered[:,0], data_filtered[:,1], label='dx_exp_filtered')
plt.legend()
plt.savefig('./_eval/' + export_filename+'_filtering_dx.png',dpi=300)


#Plot process window: raw and filtered dy
plt.figure()
plt.title('Roh + gefilterte dy Daten')
plt.plot(data[:,0], data[:,2], label='dy_exp_raw')
plt.plot(data_filtered[:,0], data_filtered[:,2], label='dy_exp_filtered')
plt.legend()
plt.savefig('./_eval/' + export_filename+'_filtering_dy.png',dpi=300)


plt.show()
