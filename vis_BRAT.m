clear all;
clf;
folder = fileparts(which('vis_BRAT')); 
addpath(genpath(folder));
grid_min = [-5; -5; -pi; 0; -1.1]; % Lower corner of computation domain
grid_max = [5; 5; pi; 0.6; 1.1];    % Upper corner of computation domain
N = [51; 51; 11; 11; 11];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
% create the grid
g = createGrid(grid_min, grid_max, N, pdDims);
% load the whole obstacle FMM
obsMap = load("obstaclemap.mat");
% load the precomputed BRAT
data = load("BRAT.mat");
% crate the level set for obstacles
obstacles = obstacle_map(g, obsMap.obs_map, obsMap.goal, obsMap.map_bounds);

% define the goal
R = 0.3;
goal = shapeCylinder(g, [3; 4; 5], [0; 0; 0; 0; 0], R);

theta_range = [-pi -pi/2 0 pi/2 pi];
v_range = [0.3];
w_range = [0];

f = figure;
f.Color = 'white';

% project the goal and the obstacles(they do not depend on the intial
% conditions)
[grid_goal, data_goal] = proj(g,goal,[0 0 1 1 1], [0 0 0]);
[grid_obs, data_obs] = proj(g,obstacles,[0 0 1 1 1], [0 0 0]);



%% print over w in the 1st dem, theta over second and v over third
p = 1; %counts the plot number
for i=1:length(w_range)
    for k=1:length(theta_range)
        for j=1:length(v_range)
            [grid_reach, data_reach] = proj(g,data.data(:,:,:,:,:,end),[0 0 1 1 1], [theta_range(k) v_range(j) w_range(i)]);
            ax = subplot(length(w_range), length(v_range)*length(theta_range), p);
            visSetIm(grid_obs, data_obs,'r');
            hold on;
            visSetIm(grid_goal, data_goal,'g');
            hold on;
            visSetIm(grid_reach, data_reach,'b');
            hold on;
            title(sprintf('(%0.1f,%0.1f,%0.1f)', theta_range(k), v_range(j), w_range(i)));

            ax.FontSize = 8;
            axis equal;
            hold off;
            p = p+1;
        end
    end
end

saveas(gcf,'BRAT_waypointNav.png')