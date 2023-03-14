from xpc3 import *
from xpc3_helper import *
from xpc3_helper_SK import *
from nnet import *
from PIL import Image

import numpy as np
import h5py
import time
import math
import mss
import cv2
import os
from tqdm import tqdm
import time
import json
import pickle

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
stride = 16             # Size of square of pixels downsampled to one grayscale value
numPix = 16             # During downsampling, average the numPix brightest pixels in each square
width  = 256//stride    # Width of downsampled grayscale image
height = 128//stride    # Height of downsampled grayscale image

screenShot = mss.mss()
mon = screenShot.monitors[1]
monitor = {
        "top": mon["top"] + 100,  # 100px from the top
        "left": mon["left"] + 100,  # 100px from the left
        "width": 1720,
        "height": 960,
    }

def get_simulator_and_model(start_params):

    TIME_OF_DAY = start_params['TIME_OF_DAY']
    CLOUD_COVER = start_params['CLOUD_COVER']
    start = start_params['start']
    ############## Start the simulation ############
    # Connect w/ X-Plane
    client = xpc3.XPlaneConnect()
    xpc3_helper.reset(client)
    time.sleep(5)

    # Set the time of the day
    client.sendDREF("sim/time/zulu_time_sec", TIME_OF_DAY*3600+8*3600) # Adding 8 to convert to UTC

    # Set the weather
    client.sendDREF("sim/weather/cloud_type[0]", CLOUD_COVER)

    # Set the start position
    # import ipdb; ipdb.set_trace()
    setHomeXYhe(client, start[0], start[1], start[2])
    time.sleep(1)
    # import ipdb; ipdb.set_trace()

    # Load the model
    model_filename = "./models/TinyTaxiNet.nnet"
    network = NNet(model_filename)
    return client, network

def save_image(img, filename=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img.astype(np.int32))
    ax.grid(False)
    if filename is None:
        fig.savefig('image.png')
    else:
        fig.savefig(filename)

def replace_image_block(img, ax1_size, ax2_size, ax1_start, ax2_start, ax1_replacement_start, ax2_replacement_start):
    img[ax1_start:ax1_start+ax1_size, ax2_start:ax2_start+ax2_size, :] = img[ax1_replacement_start:ax1_replacement_start+ax1_size, ax2_replacement_start:ax2_replacement_start+ax2_size, :]
    return img


def subplot2(plt, Y_X, sz_y_sz_x=(10, 10), space_y_x=(0.1, 0.1), T=False):
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    hspace, wspace = space_y_x
    plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
    fig, axes = plt.subplots(Y, X, squeeze=False)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if T:
        axes_list = axes.T.ravel()[::-1].tolist()
    else:
        axes_list = axes.ravel()[::-1].tolist()
    return fig, axes, axes_list

def render(ax, pos_3, plot_quiver=False, plot_fov=False, **kwargs):
    ax.plot(pos_3[1], pos_3[0], **kwargs, zorder=20)
    if plot_quiver:
        ax.quiver([pos_3[1]], [pos_3[0]],
                    np.sin(np.deg2rad(pos_3[2])), np.cos(np.deg2rad(pos_3[2])),zorder=19)
        if plot_fov:
            x_1 = 3*np.cos(pos_3[2]+np.pi/4) + pos_3[0]
            y_1 = 3*np.sin(pos_3[2]+np.pi/4) + pos_3[1]
            x_values = [x_1, pos_3[0]]
            y_values = [y_1, pos_3[1]]
            ax.plot(x_values, y_values, 'k', linestyle="--")

            x_2 = 3*np.cos(pos_3[2]-np.pi/4) + pos_3[0]
            y_2 = 3*np.sin(pos_3[2]-np.pi/4) + pos_3[1]
            x_values = [x_2, pos_3[0]]
            y_values = [y_2, pos_3[1]]
            ax.plot(x_values, y_values, 'g', linestyle="--")

def render_simulation(dir,data,f):
    """
    creates a 3 column subfigure that contains the
    position of the robot at each timestep on the trajectory
    image seen at that timestep, FOV.
    """
    freq = f # in steps
    total_steps = data['num_steps'] 
    steps = np.arange(0,total_steps,freq)
    num_steps = len(steps)
    if steps.size == 0:
        num_steps = 1
        steps = [0,1]
    fig, axss, _ = subplot2(plt, (num_steps, 5), (8, 8), (.4, .4))

    print('Rendering Results......')
    for i,t in tqdm(enumerate(steps)): 
        # render the whole traj in axss[i][0] with the start and goal and the bounding lines
        # axss[i][0].plot(data['ys'][0:-1],data['pred_xs'],color='b')
        # render the predicted traj
        if data['result'] == 1:
            axss[i][0].plot(-data['xs'][0:-1],data['ys'][0:-1],color='g',linewidth=3)
        else:
            axss[i][0].plot(-data['xs'][0:-1],data['ys'][0:-1],color='r',linewidth=3,zorder=4)

        axss[i][0].axvline(x=10, color='k', linestyle='-',linewidth=8, zorder=1)
        axss[i][0].axvline(x=-10, color='k', linestyle='-',linewidth=8, zorder=2)
        axss[i][0].axvline(x=0, color='k', linestyle='--',linewidth=5, zorder=3)
        # plot start
        render(axss[i][0], (data['ys'][0],-data['xs'][0],-data['thetas'][0]), marker='o', color='blue')
        
        # over lay the curnt position 
        render(axss[i][0], (data['ys'][t],-data['xs'][t],-data['thetas'][t]), marker='o', color='cyan', plot_quiver=True, plot_fov=False)
        # render(axss[i][0], (data['ys'][t],data['xs'][t],data['thetas'][t]), marker='o', color='yellow', plot_quiver=True, plot_fov=False)
        xabs_max = abs(max(axss[i][0].get_xlim(), key=abs))
        axss[i][0].set_xlim(xmin=-xabs_max, xmax=xabs_max)
        # over lay the predicted position 
        render(axss[i][0], (data['ys'][t],-data['pred_xs'][t],-data['pred_thetas'][t]), marker='o', color='red', plot_quiver=True, plot_fov=False)

    #   label plot with waypoint and current pos
        axss[i][0].set_xlabel('Position: [{:.2f}, {:.2f}, {:.2f}] \n Prediction: [{:.2f}, {:.2f}] \n Control: {:.2f}'.format(data['xs'][t],data['thetas'][t],data['ys'][t],data['pred_xs'][t],data['pred_thetas'][t],data['control'][t]))
        axss[i][0].set_title('Step: {:.2f}'.format(t))
       
        axss[i][1].imshow(data['orig_images'][t]/255.)
        axss[i][2].imshow(data['predicted_images'][t]/255.)
        axss[i][3].imshow(data['orig_images'][t]/255.)
        axss[i][4].imshow(data['predicted_images'][t]/255.)

        temp_1 = np.resize(data['downsampled_images'][t], (height, width))
        temp_2 = np.resize(data['predicted_downsampled_images'][t], (height, width))
        axss[i][3].imshow(temp_1, cmap='gray')
        axss[i][4].imshow(temp_2, cmap='gray')

        axss[i][1].grid(False)
        axss[i][1].axis('off')
        axss[i][2].grid(False)
        axss[i][2].axis('off')        
        axss[i][3].grid(False)
        axss[i][3].axis('off')       
        axss[i][4].grid(False)
        axss[i][4].axis('off')
        # axss[i][2].grid(False)

    fig.savefig(dir)


def getCurrentImage(iter=None):
    # time.sleep(1)
    screen_width = 360  # For cropping
    screen_height = 200  # For cropping
    # Get current screenshot
    img = cv2.cvtColor(np.array(screenShot.grab(monitor)), cv2.COLOR_BGRA2BGR)[230:,:,:]
    img = cv2.resize(img,(screen_width,screen_height))
    img = img[:,:,::-1]
    img = np.array(img)
    orig = np.copy(img)

    # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so 
    # values range between 0 and 1
    img = np.array(Image.fromarray(img).convert('L').crop((55, 5, 360, 135)).resize((256, 128)))/255.0
    # cv2.imwrite(foldername+'/'+str(iter)+'.png',orig)
    # Downsample image
    # Split image into stride x stride boxes, average numPix brightest pixels in that box
    # As a result, img2 has one value for every box
    img2 = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            img2[i,j] = np.mean(np.sort(img[stride*i:stride*(i+1),stride*j:stride*(j+1)].reshape(-1))[-numPix:])

    # Ensure that the mean of the image is 0.5 and that values range between 0 and 1
    # The training data only contains images from sunny, 9am conditions.
    # Biasing the image helps the network generalize to different lighting conditions (cloudy, noon, etc)
    img2 -= img2.mean()
    img2 += 0.5
    img2[img2>1] = 1
    img2[img2<0] = 0
    return orig, img2.flatten()

def dynamics(x, y, theta, phi_deg, dt = 0.05, v = 5, L = 5):
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi_deg)

    x_dot = v * np.sin(theta_rad)
    y_dot = v * np.cos(theta_rad)
    theta_dot = (v / L) * np.tan(phi_rad)

    x_prime = x + x_dot * dt
    y_prime = y + y_dot * dt
    theta_prime = theta + np.rad2deg(theta_dot) * dt

    return x_prime, theta_prime, y_prime


def rollout(client, network, num_steps=500, dt=0.05, ctrl_every=1, sim_dynamics='dubins', ideal=0, image_getter = getCurrentImage):
    cte, he = getErrors(client)
    _, dtp = getHomeXY(client)

    print("Performing Rollout Simulation...\n")

    phi_deg = 0.0

    num_control = int(num_steps / ctrl_every - 1)

    xs = np.zeros(num_steps + 1)
    pred_xs = np.zeros(num_control + 1)
    ys = np.zeros(num_steps + 1)
    thetas = np.zeros(num_steps + 1)
    pred_thetas = np.zeros(num_control + 1)
    control = np.zeros(num_control + 1)

    orig_images = np.zeros((num_steps + 1, 200, 360, 3))
    downsampled_images = np.zeros((num_steps + 1, 128))

    orig_images_pred = np.zeros((num_steps + 1, 200, 360, 3))
    downsampled_images_pred = np.zeros((num_steps + 1, 128))

    xs[0] = cte
    ys[0] = dtp
    thetas[0] = he
    result = 1
    episode_steps = -1
    for i in range(num_steps):
        
        orig, image = image_getter(i)
        if i % ctrl_every == 0:
            # Get network prediction and control
            pred = network.evaluate_network(image)
            cte_pred = pred[0]
            he_pred = pred[1]

            pred_xs[int(i / ctrl_every)] = cte_pred
            pred_thetas[int(i / ctrl_every)] = he_pred

            if ideal:
                phi_deg = -0.74 * xs[i] - 0.44 * thetas[i] # Steering angle from ideal
            else:
                phi_deg = -0.74 * cte_pred - 0.44 * he_pred # Steering angle from nncontroller
            # 
            # return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 
            control[int(i / ctrl_every)] = phi_deg

        # find the image at the predicted position
        # setHomeXYhe(client, cte_pred, dtp, he_pred)
        # time.sleep(1)
        orig_pred, image_pred = image_getter(i)

        # Get next states and go to them
        if sim_dynamics == 'dubins':
            # Simulate dubins dynamics
            cte, he, dtp = dynamics(cte, dtp, he, phi_deg, dt)
            setHomeXYhe(client, cte, dtp, he)
            # time.sleep(1)
        else:
            # Actually simulate the X-plane
            raise NotImplementedError
        

        print("Step ", i+1, " of ", num_steps, ". cte: ", cte,  "dtp: ", dtp, "he: ", he)
            # import ipdb; ipdb.set_trace()

        xs[i + 1] = cte
        ys[i + 1] = dtp
        thetas[i + 1] = he
        orig_images[i, :, :, :] = orig
        downsampled_images[i, :] = image

        orig_images_pred[i, :, :, :] = orig_pred
        downsampled_images_pred[i, :] = image_pred

        if abs(cte) > 10.0:
            print('System entered the unsafe state')
            result = 0
            episode_steps = i

        time.sleep(0.01)
    return xs, ys, thetas, pred_xs, pred_thetas, control, orig_images, downsampled_images, result, episode_steps, orig_images_pred, downsampled_images_pred

def save_metrics(xs, ys, thetas, pred_xs, pred_thetas, control, orig_images, downsampled_images, result, episode_steps, start_params, orig_images_pred=None, downsampled_images_pred=None):
    TIME_OF_DAY = start_params['TIME_OF_DAY']
    CLOUD_COVER = start_params['CLOUD_COVER']
    start = start_params['start']
    foldername = start_params['foldername']
    ideal = start_params['ideal']
    filename = foldername
    if os.path.exists(foldername):
        t = time.localtime(time.time())
        foldername = foldername+'_'+str(t[1])+'_'+str(t[2])+'_'+str(t[3])+'_'+str(t[4])+'_'+str(t[5])
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filename = foldername+'/'+filename
    data = {'num_steps': start_params['num_steps'],
            'ctrl_every': start_params['control_every'],
            'sim_type': start_params['sim_type'],
            'start': (start[0], start[1], start[2]),
            'time': TIME_OF_DAY,
            "cloud": CLOUD_COVER,
            'controller': "ideal" if ideal else "NN",
            'control': control,
            'freq': start_params['freq'],
            'xs': xs,
            'ys': ys, 
            'thetas': thetas, 
            'pred_xs': pred_xs, 
            'pred_thetas': pred_thetas, 
            'orig_images': orig_images, 
            'downsampled_images': downsampled_images,
            'predicted_images': orig_images_pred,
            'predicted_downsampled_images': downsampled_images_pred,
            'result': result,
            'episode_steps': episode_steps}

    with open(filename+".pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    keys = ['ctrl_every', 'sim_type', 'start', 'time', "cloud", 'controller','freq','result']
    dict2 = {x:data[x] for x in keys}
    with open(filename+'.json', 'w') as f:
        json.dump(dict2, f)
    
    # # save the images seen
    # os.makedirs(foldername+'/'+'imgs')
    # for step in range(start_params['num_steps']):
    #     plt.imshow(orig_images[step]/255.0)
    #     plt.imsave(foldername+'/'+'imgs/'+str(step)+'.png',orig_images[step]/255.0)
    #     plt.close()

    render_simulation(filename+".pdf",data,start_params['freq'])