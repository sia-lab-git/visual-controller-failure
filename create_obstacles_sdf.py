'''
Create obstacle.mat file from the traversibles
obstacle.mat contains the SDF from the environmental obstacles.
The traversible is an occupany map of the environment
obstacle.map will be used to initialize the obsacles in the 
BRAT computation. 
'''
from scipy.io import savemat
import argparse
from Visual_Navigation_Release.params.base_data_directory import *
import numpy as np
import sys
import skfmm
import pickle

parser = argparse.ArgumentParser(description='SDF generator')
parser.add_argument('--building_number', type=int, 
                    default='1',
                    help='The building for which to create the SDF')
parser.add_argument('--goal', type=float, nargs='+', default=[15,25],
                    help='goal location (default: [15,25])')

args = parser.parse_args()

file_name = base_data_dir() + "/stanford_building_parser_dataset/traversibles/area{:d}/data.pkl".format(args.building_number)
goal = args.goal

try:
    with open(file_name, 'rb') as f:
        x = pickle.load(f)
except:
    print("Incorrect or traversible file name not provided")
    sys.exit()

obs_signed = np.copy(x['traversible'])
obs_signed = -1*obs_signed
obs_signed[obs_signed == -0.] = 1
implicit_surface = skfmm.distance(obs_signed,dx=[0.05,0.05])

map_bounds = obs_signed.shape # represents the area of the map in pixels
map_bounds = np.array(map_bounds)*0.05 # represents the physical area of the map in meters

mdic = {"obs_map": implicit_surface, "map_bounds":map_bounds, "goal":args.goal }
savemat("obstaclemap.mat", mdic)