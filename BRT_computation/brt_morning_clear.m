clear; clc; close all;


% specify the location where the BRT will be saved
datafilename = './data/brt_morning_clear.mat'; % output

% specify the location where the controller data is found
controllerfilename = '../data/morning_clear/controller.mat'; % input

save_data = 1;

%% Load the contrller
load(controllerfilename);
clear g;

%% Grid for the controller
grid_min = [-11; 100; -30*pi/180]; % Lower corner of computation domain
grid_max = [11; 250; 30*pi/180];    % Upper corner of computation domain
N = [41; 101; 41];         % Number of grid points per dimension
gCtrl = createGrid(grid_min, grid_max, N);

%% Grid for the computation
grid_min = [-11; 100; -32*pi/180]; % Lower corner of computation domain
grid_max = [11; 250; 32*pi/180];    % Upper corner of computation domain
% N = [41; 101; 41];         % Number of grid points per dimension
N = [101; 101; 101];         % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N);

%% Reshape the controller
% Compute the control commands
controller = reshape(controller, [gCtrl.N', 2]);
gainCTE = -0.74;
gainHE = -0.44;
u = gainCTE * controller(:, :, :, 1) + gainHE * controller(:, :, :, 2); % Phi in degrees
u = u * pi/180; % Phi in radians
controller = tan(u);

% Interpolate if required
gflat = cat(2, g.xs{1}(:), g.xs{2}(:), g.xs{3}(:));
controller = eval_u(gCtrl, controller, gflat);
controller = reshape(controller, g.N');

%% Target set
% data0 = -shapeRectangleByCorners(g, [-10; grid_min(2); grid_min(3)], [10; grid_max(2); grid_max(3)]);
data0 = -shapeRectangleByCorners(g, [-10; -inf; -inf], [10; inf; inf]);

%% time vector
t0 = 0;
tMax = 8.0;
dt = 0.1;
tau = [t0:dt:tMax];

%% Pack problem parameters
dCar = TaxiNet3D([0, 0, 0], gainCTE, gainHE, controller, 5.0, 5.0);
schemeData.grid = g;
schemeData.dynSys = dCar;
schemeData.uMode = 'max';
schemeData.accuracy = 'low';

%% Compute BRS
extraArgs.visualize.valueSet = true;
extraArgs.visualize.initialValueSet = 1;
extraArgs.visualize.figNum = 2; %set figure number
extraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update

extraArgs.visualize.plotData.plotDims = [1 0 1];
extraArgs.visualize.plotData.projpt = 'min';
extraArgs.visualize.viewAngle = [0,90]; % view 2D

extraArgs.stopConverge = true;
extraArgs.convergeThreshold = 0.001;

numPlots = 4;
spC = ceil(sqrt(numPlots));
spR = ceil(numPlots / spC);

[data, tau, ~] = HJIPDE_solve(data0, tau, schemeData, 'zero', extraArgs);

%% Save the data
if save_data == 1
  save(datafilename, 'g', 'data', 'tau');
end
