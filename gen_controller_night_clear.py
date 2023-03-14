# Script to generate the controller data.
# Step 1: Kick off the correct aircraft and terminal.
# Step 2: Setup the taxinet camera
# Step 3: Setup the correct monitor and resolution.

from xpc3 import *
from xpc3_helper import *
from xpc3_helper_SK import *
from nnet import *
from simulator_utils import *
from PIL import Image

import numpy as np
import h5py
import time
import mss
import cv2
import os
import matplotlib.pyplot as plt

import pickle

import scipy.io as spio

########### Set PARAMS here ############

# Set to a random initial state
start_cte = 8.0 
start_dtp = 100.0 
start_he = 0 

TIME_OF_DAY = 21.0 # military time
CLOUD_COVER = 0  #0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast

##########################################

# Folder where the results and data will be saved
logging_root = "./data/night_clear"
if not os.path.exists(logging_root):
    os.makedirs(logging_root)
start_params = {'foldername': "",
                'save_traj': 0,
                'ideal': 0,
                'freq': 0,
                'TIME_OF_DAY': TIME_OF_DAY,
                'CLOUD_COVER': CLOUD_COVER,
                'start': [start_cte,start_dtp,start_he],
                'dt': 0,
                'tMax': 0,
                'num_steps': 0,
                'control_every': 1,
                'sim_type': 'dubins'
                }


client, network = get_simulator_and_model(start_params)

# Load the grid in x, y, theta order. Theta should be in degrees. 
grid_filename = 'brt_computation_grid.mat'
grid = spio.loadmat(grid_filename)
grid = grid['gmat']
N = grid.shape[0]
 
# Initialize the controller
controller = -200*np.ones((N, 2))

orig_images = []
downsampled_images = []

for i in range(N):
    # t1 = time.time()
    print('State %i of %i' %(i+1, N))

    # Set the client at the current state
    setHomeXYhe(client, grid[i, 0], grid[i, 1], grid[i, 2])
    time.sleep(0.01)

    # Obtain the image and the downsampled image at the current state
    _, image = getCurrentImage()

    # Find the prediction of the CNN 
    pred = network.evaluate_network(image)
    controller[i] = pred

# Save the controller
data_dict = {}
data_dict['g'] = grid
data_dict['controller'] = controller
spio.savemat(os.path.join(logging_root, 'controller.mat'), data_dict)