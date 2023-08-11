%% Grid
folder = fileparts(which('compute_BRT')); 
addpath(genpath(folder));
grid_min = [-5; -5; -pi; 0; -1.1]; % Lower corner of computation domain
grid_max = [5; 5; pi; 0.6; 1.1];    % Upper corner of computation domain
N = [51; 51; 11; 11; 11];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

%% target set- this is where we define the target as an implicit surface function
R = .3;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
data0 = shapeCylinder(g, [3; 4; 5], [0; 0; 0; 0; 0], R);
% data0 = shapeSphere(g,[0;0;0],2);
% also try shapeRectangleByCorners, shapeSphere, etc.

%% time vector
t0 = 0;
tMax = 100;
dt = 1;
tau = t0:dt:tMax;

%% problem parameters

% input bounds
% control trying to min or max value function?
uMode = 'min';
% do dStep2 here
dMode = 'max';
%% Pack problem parameters

% Define dynamic system
nncontrol = load('optCtrl.mat');
robot = Robot5D([0, 0, 0, 0, 0], nncontrol.speed, nncontrol.ang_vel, nncontrol.acc, nncontrol.ang_acc);

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = robot;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = 'min';
schemeData.dMode = 'max';

%% If you have obstacles, compute them here
obsMap = load("obstaclemap.mat");
obstacles = obstacle_map(g, obsMap.obs_map, obsMap.goal, obsMap.map_bounds );
HJIextraArgs.obstacles = obstacles;

%% Compute value function

HJIextraArgs.visualize.valueSet = 1;
HJIextraArgs.visualize.initialValueSet = 1;
HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = false; %delete previous plot as you update


% Plot a 2D slice
HJIextraArgs.visualize.plotData.plotDims = [1 1 0 0 0]; %plot x, y
HJIextraArgs.visualize.plotData.projpt = [0 0 0]; %project at theta = 0
HJIextraArgs.visualize.viewAngle = [0,90]; % view 2D

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
[data, tau2, ~] = HJIPDE_solve(data0, tau, schemeData, 'minVOverTime', HJIextraArgs);
save('BRAT.mat','data','-v7.3','-nocompression');