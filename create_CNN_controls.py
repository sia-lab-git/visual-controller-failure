'''
Dummy file that can render images at random poses
all other configs remain the same as the paper
'''

import os
import sys
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from utils import utils
import pickle
import argparse
os.environ["PYOPENGL_PLATFORM"] = "egl"
import matplotlib.pyplot as plt
import glob
from scipy import io
import importlib
import datetime
from training_utils.trainer_helper import TrainerHelper
from trajectory.trajectory import SystemConfig, Trajectory
from models.visual_navigation.rgb.resnet50.rgb_resnet50_waypoint_model import RGBResnet50WaypointModel
from planners.nn_waypoint_planner_batch import NNWaypointPlanner

# https://github.com/google/jax/issues/4920
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def create_params(param_file):
    spec = importlib.util.spec_from_file_location('parameter_loader', param_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo.create_params() 

def create_session_dir(p):
    job_dir = "reproduce_LB_WayptNavResults"
    trainer_dir = p.trainer.ckpt_path.split('checkpoints')[0]
    checkpoint_number = int(p.trainer.ckpt_path.split('checkpoints')[1].split('-')[1])
    job_dir = os.path.join(trainer_dir, 'test', 'checkpoint_{:d}'.format(checkpoint_number), job_dir)
    utils.mkdir_if_missing(job_dir)
    p.job_dir = job_dir
    p.session_dir = os.path.join(p.job_dir,
                                          'session_%s' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

def parse_params(p):
        """
        Parse the parameters based on args.command
        to add some additional helpful parameters.
        """
        p.simulator_params = p.test.simulator_params

        # Parse the dependencies
        p.simulator_params.simulator.parse_params(p.simulator_params)
        return p

def configure_plotting():
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

def nn_simulator_params(p):
    """
    Returns a DotMap object with simulator parameters
    for a simulator which uses a NN based planner
    """
    from copy import deepcopy
    # Create an input and output model
    model = RGBResnet50WaypointModel(p)

    # Create a trainer
    trainer = TrainerHelper(p)

    # Load the checkpoint
    trainer.restore_checkpoint(model=model)
    
    p = deepcopy(p.simulator_params)
    p.planner_params.planner = NNWaypointPlanner
    p.planner_params.model = model
    p.simulator_name = 'RGB_Resnet50_NN_Waypoint_Simulator'
    return p

def expert_simulator_params(p):
    from copy import deepcopy
    p = deepcopy(p.simulator_params)
    p.simulator_name = 'Expert_Simulator'
    return p

def nn_simulator_params_render(p):
    """
    Returns a DotMap object with simulator parameters
    for a simulator which uses a NN based planner
    """
    from copy import deepcopy
    p = deepcopy(p.simulator_params)
    p.simulator_name = 'Render_Simulator'
    # Do not downscale image if just simulator
    p.obstacle_map_params.renderer_params.camera_params.width = 1080
    p.obstacle_map_params.renderer_params.camera_params.height = 1080
    p.obstacle_map_params.renderer_params.camera_params.im_resize = 1
    return p

def init_simulator_data(p, num_tests, seed, name='', dirname='', plot_controls=False,
                             base_dir=None):
        """Initializes a simulator_data dictionary based on the params in p,
        num_test, name, and dirname. This can be later passed to the simulate
        function to test a simulator."""
        # Parse the simulator params
        p.simulator.parse_params(p)
        
        # change the bacth number
        p.planner_params.batch = num_tests

        # Initialize the simulator
        simulator = p.simulator(p)

        # Create Figures/ Axes
        if plot_controls:
            # Each row has 2 more subplots for linear and angular velocity respectively
            fig, axss, _ = utils.subplot2(plt, (num_tests, 3), (8, 8), (.4, .4))
        else:
            fig, axss, _ = utils.subplot2(plt, (num_tests, 1), (8, 8), (.4, .4))

        if base_dir is None:
            base_dir = p.session_dir

        # Construct data dictionray
        simulator_data = {'name': name,
                          'simulator': simulator,
                          'fig': fig,
                          'axss': axss,
                          'dir': dirname,
                          'n': num_tests,
                          'seed': seed,
                          'base_dir': base_dir}

        return simulator_data

def restore_checkpoint(self, model):
    """
    Load a given checkpoint.
    """
    # Create a checkpoint
    self.checkpoint = tf.Checkpoint(optimizer=self.create_optimizer(), model=model.arch)
    
    # Restore the checkpoint
    self.checkpoint.restore(self.p.ckpt_path).expect_partial()

def create_dataset(batch_size=1, goal=[0,0], size=5, num_points=(51,51,11,11,11)):
    x_range = np.linspace(goal[0]-size,goal[0]+size,num_points[0])
    y_range = np.linspace(goal[1]-size,goal[1]+size,num_points[1])
    theta_range = np.linspace(-np.pi,np.pi,num_points[2],False)
    speed_range = np.linspace(0,0.6,num_points[3])
    ang_speed_range = np.linspace(-1.1,1.1,num_points[4])
    current_size = 0
    batch = []
    dataset = []
    for x in x_range:
        for y in y_range:
            for theta in theta_range:
                for speed in speed_range:
                    for ang_speed in ang_speed_range:
                        batch.append([x,y,theta,speed,ang_speed])
                        current_size += 1
                        if current_size == batch_size:
                            dataset.append(batch)
                            batch = []
                            current_size = 0
    dataset = np.array(dataset)
    return dataset, goal

def create_dataset_v2(batch_size=1, goal=[0,0], preloaded_dataset = None):
    preloaded_dataset = np.load(preloaded_dataset)
    current_size = 0
    batch = []
    dataset = []
    for state_number in range(preloaded_dataset.shape[0]):
        state = preloaded_dataset[state_number,:] # this should be a 1X5
        batch.append(state)
        current_size += 1
        if current_size == batch_size:
            dataset.append(batch)
            batch = []
            current_size = 0
    dataset = np.array(dataset)
    return dataset, goal

def reset_batch(simulator, seed=-1, n=1, batch_num=1, goal=[0,0], dataset=None):
    '''
    modified version of the simulator reset
    here we reset a batch of starts and one single goal
    other initial conditions are zero
    '''
    if seed != -1:
        simulator.rng.seed(seed)

    # Note: Obstacle map must be reset independently of the fmm map.
    # Sampling start and goal may depend on the updated state of the
    # obstacle map. Updating the fmm map depends on the newly sampled goal.
    simulator._reset_obstacle_map(simulator.rng)
    _reset_start_configuration_batch(simulator, n=n, batch_num=batch_num, dataset=dataset)
    _reset_goal_configuration_batch(simulator, n=n, goal=goal)
    simulator._update_fmm_map()

    simulator.vehicle_trajectory = Trajectory(dt=simulator.params.dt, n=n, k=0)
    simulator.obj_val = np.inf
    simulator.vehicle_data = {}

def reset_velocity(simulator, start, seed=-1, n=1):
    '''
    modified version of the simulator reset
    here we reset a batch of differnet starting velocity for the same starting and goal position
    starts and one single goal
    other initial conditions are zero
    '''
    if seed != -1:
        simulator.rng.seed(seed)

    # Note: Obstacle map must be reset independently of the fmm map.
    # Sampling start and goal may depend on the updated state of the
    # obstacle map. Updating the fmm map depends on the newly sampled goal.
    simulator._reset_obstacle_map(simulator.rng)

    start_n2 = np.tile(np.array([start]),(n,1))
    start_n2 = start_n2.astype(np.float32)
    start_n12 = np.expand_dims(start_n2, axis=1)

    heading_11 = np.array([[0]],dtype=np.float32)
    heading_n1 = np.tile(heading_11,(n,1))
    heading_n11 = np.expand_dims(heading_n1, axis=1)

    # speed = np.linspace(0,0.6,n)
    # speed_n11 = speed.reshape((n, 1, 1))

    ang_speed = np.linspace(-1.1,1.1,n)
    ang_speed_n11 = ang_speed.reshape((n, 1, 1))

    # zero ang speed
    # ang_speed_11 = np.array([[0]],dtype=np.float32)
    # ang_speed_n1 = np.tile(ang_speed_11,(n,1))
    # ang_speed_n11 = np.expand_dims(ang_speed_n1, axis=1)

    # zero speed
    speed_11 = np.array([[0]],dtype=np.float32)
    speed_n1 = np.tile(speed_11,(n,1))
    speed_n11 = np.expand_dims(speed_n1, axis=1)

    simulator.start_config = SystemConfig(dt=p.dt, n=n, k=1,
                                        position_nk2=start_n12,
                                        heading_nk1=heading_n11,
                                        speed_nk1=speed_n11,
                                        angular_speed_nk1=ang_speed_n11)

    # The system dynamics may need the current starting position for
    # coordinate transforms (i.e. realistic simulation)
    simulator.system_dynamics.reset_start_state(simulator.start_config)

    goal_12 = np.array([[15, 25]],dtype=np.float32)
    goal_n2 = np.tile(goal_12,(n,1))
    goal_n12 = np.expand_dims(goal_n2, axis=1)


    # Initialize the goal configuration
    simulator.goal_config = SystemConfig(dt=p.dt, n=n, k=1,
                                    position_nk2=goal_n12)

    simulator._update_fmm_map()

    simulator.vehicle_trajectory = Trajectory(dt=simulator.params.dt, n=n, k=0)
    simulator.obj_val = np.inf
    simulator.vehicle_data = {}

def reset_custom(simulator, start, goal,  seed=-1, n=1):
    '''
    modified version of the simulator reset
    here we reset a batch of starts and one single goal
    other initial conditions are zero
    '''
    if seed != -1:
        simulator.rng.seed(seed)

    # Note: Obstacle map must be reset independently of the fmm map.
    # Sampling start and goal may depend on the updated state of the
    # obstacle map. Updating the fmm map depends on the newly sampled goal.
    simulator._reset_obstacle_map(simulator.rng)

    start_n2 = np.array([[start[0], start[1]]])
    start_n2 = start_n2.astype(np.float32)
    start_n12 = np.expand_dims(start_n2, axis=1)

    heading_n1 = np.array([[start[2]]])
    heading_n1 = heading_n1.astype(np.float32)
    heading_n11 = np.expand_dims(heading_n1, axis=1)

    speed_n11 = np.array([[[start[3]]]])
    speed_n11 = speed_n11.astype(np.float32)

    ang_speed_n11 = np.array([[[start[4]]]])
    ang_speed_n11 = ang_speed_n11.astype(np.float32)


    simulator.start_config = SystemConfig(dt=p.dt, n=n, k=1,
                                        position_nk2=start_n12,
                                        heading_nk1=heading_n11,
                                        speed_nk1=speed_n11,
                                        angular_speed_nk1=ang_speed_n11)

    # The system dynamics may need the current starting position for
    # coordinate transforms (i.e. realistic simulation)
    simulator.system_dynamics.reset_start_state(simulator.start_config)

    goal_12 = np.array([[goal[0], goal[1]]],dtype=np.float32)
    goal_n2 = np.tile(goal_12,(n,1))
    goal_n12 = np.expand_dims(goal_n2, axis=1)


    # Initialize the goal configuration
    simulator.goal_config = SystemConfig(dt=p.dt, n=n, k=1,
                                    position_nk2=goal_n12)

    simulator._update_fmm_map()

    simulator.vehicle_trajectory = Trajectory(dt=simulator.params.dt, n=n, k=0)
    simulator.obj_val = np.inf
    simulator.vehicle_data = {}

# TODO reimplement this
def _reset_start_configuration_batch(simulator, n=1, batch_num=1, dataset=None):
    """
    Reset the starting configuration of the vehicle. I only added for custom and not for the random
    """
    p = simulator.params.reset_params.start_config

    # Initialize the start configuration
    start_n2 = dataset[batch_num][:,:2]
    start_n2 = start_n2.astype(np.float32)
    start_n12 = np.expand_dims(start_n2, axis=1)

    heading_n1 = dataset[batch_num][:,2:3]
    heading_n1 = heading_n1.astype(np.float32)
    heading_n11 = np.expand_dims(heading_n1, axis=1)

    speed_n1 = dataset[batch_num][:,3:4]
    speed_n1 = speed_n1.astype(np.float32)
    speed_n11 = np.expand_dims(speed_n1, axis=1)

    ang_speed_n1 = dataset[batch_num][:,4:5]
    ang_speed_n1 = ang_speed_n1.astype(np.float32)
    ang_speed_n11 = np.expand_dims(ang_speed_n1, axis=1)

    # ang_speed_n11 = np.zeros((n, 1, 1))
    # speed_n11 = np.zeros((n, 1, 1))
    simulator.start_config = SystemConfig(dt=p.dt, n=n, k=1,
                                        position_nk2=start_n12,
                                        heading_nk1=heading_n11,
                                        speed_nk1=speed_n11,
                                        angular_speed_nk1=ang_speed_n11)

    # The system dynamics may need the current starting position for
    # coordinate transforms (i.e. realistic simulation)
    simulator.system_dynamics.reset_start_state(simulator.start_config)


def _reset_goal_configuration_batch(simulator, n=1,goal=[0,0]):
    p = simulator.params.reset_params.goal_config
    goal_12 = np.array([[goal[0], goal[1]]],dtype=np.float32)
    goal_n2 = np.tile(goal_12,(n,1))
    goal_n12 = np.expand_dims(goal_n2, axis=1)


    # Initialize the goal configuration
    simulator.goal_config = SystemConfig(dt=p.dt, n=n, k=1,
                                    position_nk2=goal_n12)

def simulate_batch(simulator):
    config = simulator.start_config
    # vehicle_trajectory = simulator.vehicle_trajectory
    # vehicle_data = simulator.planner.empty_data_dict()
    planner_data = simulator.planner.optimize(config)
    return planner_data

def create_mat(folder_name = "", goal=[15,25,0,0,0], disc=[51,51,11,11,11],batch_size=11):
    files = glob.glob(folder_name+"/*.pkl")
    goal = np.tile(np.array([goal]),(batch_size,1))
    large_egopose_nplus15 = np.zeros((1,5))
    large_action_nplus14 = np.zeros((1,4))
    print("creating the matlab consumable controllers.............................")
    for i in tqdm(range(len(files))):
        with open(files[i], 'rb') as f:
            x = pickle.load(f)
        dt = x["trajectory"]["dt"]
        pos_n2 = x['start_config']['position_nk2'][:,0,:]
        orientation_n1 = x['start_config']['heading_nk1'][:,0,:]
        speed_n1 = x['start_config']['speed_nk1'][:,0]
        ang_speed_n1 = x['start_config']['angular_speed_nk1'][:,0]
        pose_n5 = np.hstack((pos_n2,orientation_n1,speed_n1,ang_speed_n1))
        egopose_n5 = pose_n5 - goal
        large_egopose_nplus13 = np.vstack((large_egopose_nplus15,egopose_n5))
        controlspeed_n1 = x['trajectory']['speed_nk1'][:,0]
        controlangvel_n1 = x['trajectory']['angular_speed_nk1'][:,0]
        controlacceleration_n1 = (controlspeed_n1 - speed_n1)/dt
        controlangacceleration_n1 = (controlangvel_n1 - ang_speed_n1)/dt
        action_n4 = np.hstack((controlspeed_n1,controlangvel_n1,controlacceleration_n1,controlangacceleration_n1))
        large_action_nplus14 = np.vstack((large_action_nplus14,action_n4))
        
    large_egopose_n3 = large_egopose_nplus13[1:,:]
    large_action_n4 = large_action_nplus14[1:,:]


    uOpt = {'speed':large_action_n4[:,0].reshape(disc),
            'ang_vel':large_action_n4[:,1].reshape(disc),
            'acc':large_action_n4[:,2].reshape(disc),
            'ang_acc':large_action_n4[:,3].reshape(disc)}

    io.savemat("../optCtrl.mat",uOpt)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='BRAT data generator')
    parser.add_argument('--param-files', type=str, 
                        default='params/rgb_trainer/reproduce_LB_WayPtNav_results/rgb_waypoint_trainer_finetune_params.py',
                        help='path of rgb_waypoint_trainer_finetune_params.py')
    parser.add_argument('--trajectory-data-dir', type=str, default='CNN_controller_trajs',
                        help='intermediate folder to save the trajectory files (default=CNN_controller_trajs)')
    parser.add_argument('--batch-size', type=int, default=11,
                        help='input batch size for training (default: 11)')
    parser.add_argument('--num-points', type=int, nargs='+', default=[51,51,11,11,11],
                        help='input batch size for testing (default: (51,51,11,11,11)))')
    parser.add_argument('--goal', type=float, nargs='+', default=[15,25],
                        help='input batch size for testing (default: [15,25])')
    parser.add_argument('--gpu', type=str, default='/gpu:1',
                        help='device type default: /gpu:0')
    args = parser.parse_args()

    ##########################################################################################
    # No need to change anything in this block 

    # point to path of rgb_waypoint_trainer_finetune_params.py
    param_files = args.param_files
    # folder to save the trajectories
    trajectory_data_dir = args.trajectory_data_dir

    # Step 0: Set the batch size. This depends on the capacity of the hardware 
    batch_size = args.batch_size  # this is the batch size
    num_points = tuple(args.num_points)
    goal = args.goal
    
    p = create_params(param_files)
    create_session_dir(p)
    parse_params(p)
    p.device = args.gpu
    configure_plotting()

    simulate_kwargs = {}
    simulator_name = 'RGB_Resnet50_NN_Waypoint_Simulator'

    utils.mkdir_if_missing(trajectory_data_dir)

    if p.test.seed != -1:
        np.random.seed(seed=p.test.seed)
        tf.random.set_seed(seed=p.test.seed)
    
    with tf.device(p.device):
        simulator_params = nn_simulator_params(p)
        simulator_data = init_simulator_data(simulator_params,
                                                batch_size,
                                                p.test.seed,
                                                name=simulator_params.simulator_name,
                                                dirname=simulator_params.simulator_name.lower(),
                                                plot_controls=False)
        simulator = simulator_data['simulator']
        batch = simulator_data['n']
        seed = simulator_data['seed']
        dirname = simulator_data['dir']
        base_dir = simulator_data['base_dir']
        kwargs = {}
    ##########################################################################################
        
        # For generating the controller data
        
        # Step 1: Create the dataset grid for the entire config space set the goal
        # currently assumes the grid to be a square patch of size 'size' centred at the goal
        # the the shape of the grid in the statepace is given by 'num_points'

        dataset, goal = create_dataset(batch_size=batch,goal=goal,size=5,num_points=num_points)

        # import pdb;pdb.set_trace()
        # Step 2: remember to set the conditions to false in nn_planner.py, nn_waypoint_planner.py, planner.py
        # Step 3: Generate the controls for the entire dataset
        print("Sampling controllers from the CNN and simulator.............................")
        for b in tqdm(range(dataset.shape[0])):
            reset_batch(simulator,seed=seed,n=batch,batch_num=b,goal=goal,dataset=dataset)
            data = simulate_batch(simulator)
            data['start_config'] = simulator.start_config.to_numpy_repr()
            trajectory_file = os.path.join(trajectory_data_dir, 'traj_{:d}.pkl'.format(b))
            with open(trajectory_file, 'wb') as f:
                pickle.dump(data, f)

    # Create the .mat file that can be comsumed by the BRAT computation
    try:
        create_mat(trajectory_data_dir,
                goal=[goal[0],goal[1],0,0,0],
                disc=num_points,
                batch_size=batch_size)
    except:
        print("error in creating the .mat file")