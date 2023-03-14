clear; clc; 


% specify the location where the BRT will be saved
datafilename = 'Data_new/true_BRT_gainCTE_0x74_gainHE_0x44.mat';

save_data = 1;

%% Grid
grid_min = [-11; -32*pi/180]; % Lower corner of computation domain
grid_max = [11; 32*pi/180];    % Upper corner of computation domain
N = [101; 101];         % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N);

%% Target set
data0 = -shapeRectangleByCorners(g, [-10, -inf], [10, inf]);

%% time vector
t0 = 0;
tMax = 500.0;
dt = 0.1;
tau = [t0:dt:tMax];

%% Pack problem parameters
% Dynamical system parameters
dCar = TaxiNet2D([0, 0], -0.74, -0.44, 5.0);
% dCar = TaxiNet2D([0, 0], 0.015, 0.008, 5.0);
schemeData.grid = g;
schemeData.dynSys = dCar;
schemeData.uMode = 'max';
schemeData.accuracy = 'low';

%% Compute BRS
extraArgs.visualize.valueSet = true;
extraArgs.visualize.initialValueSet = 1;
extraArgs.visualize.figNum = 1; %set figure number
extraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update

extraArgs.stopConverge = true;
extraArgs.convergeThreshold = 0.0001;

numPlots = 4;
spC = ceil(sqrt(numPlots));
spR = ceil(numPlots / spC);

[data, tau, ~] = HJIPDE_solve(data0, tau, schemeData, 'zero', extraArgs);

%% Save the data
if save_data == 1
  save(datafilename, 'g', 'data', 'tau');
end
