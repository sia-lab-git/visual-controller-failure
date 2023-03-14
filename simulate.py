from simulator_utils import *

########### Set PARAMS here ############
start_cte = -5.5  # -6 - fail
start_dtp = 110.0 # 110 - fail; 130 - safe.
start_he = -17  # -17 fail

foldername = 'simulation'
save_traj = 0
ideal = 0 # 1 for ideal controller 0 for NN-controller
freq = 1
TIME_OF_DAY = 9.0 # military time
CLOUD_COVER = 0  #0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast


dt = 0.05
tMax = 10


##########################################
num_steps = int(tMax/dt)
start_params = {'foldername': foldername,
                'save_traj': save_traj,
                'ideal': ideal,
                'freq': freq,
                'TIME_OF_DAY': TIME_OF_DAY,
                'CLOUD_COVER': CLOUD_COVER,
                'start': [start_cte,start_dtp,start_he],
                'dt': dt,
                'tMax': tMax,
                'num_steps': num_steps,
                'control_every': 1,
                'sim_type': 'dubins'
                }


client, network = get_simulator_and_model(start_params)


# Start the simulation
xs, ys, thetas, pred_xs, pred_thetas, control, orig_images, downsampled_images, result, episode_steps, predicted_images, predicted_downsampled_images = rollout(client, network, 
                                                                                                                num_steps = num_steps, 
                                                                                                                dt = dt, 
                                                                                                                ideal=ideal)

if save_traj:
    save_metrics(xs, ys, thetas, pred_xs, pred_thetas, control, orig_images, downsampled_images, result, episode_steps, start_params, predicted_images, predicted_downsampled_images)
