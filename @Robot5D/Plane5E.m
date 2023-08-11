classdef Plane5E < DynSys
  properties
    % Angular control bounds
%     alphaMax
    
    % Acceleration control bounds
%     aRange
    
    % Disturbance bounds
%     dMax
    
    % Active dimensions
    dims

    % optimal nn controls
    speed
    ang_vel
    acc
    ang_acc
  end
  
  methods
    function obj = Plane5E(x, u1, u2, u3, u4)
      % obj = Plane5D(x, wMax, aRange, dMax, dims)
      %     Dynamics of the Plane5D
      %         \dot{x}_1 = x_4 * cos(x_3) + d_1 (x position)
      %         \dot{x}_2 = x_4 * sin(x_3) + d_2 (y position)
      %         \dot{x}_3 = x_5                  (heading)
      %         \dot{x}_4 = u_3 + d_3            (linear speed)
      %         \dot{x}_5 = u_4 + d_4            (turn rate)
      %           aRange(1) <= u_1 <= aRange(2)
      %           -alphaMax <= u_2 <= alphaMax
      
%       if nargin < 4
%         dMax = [0; 0; 0; 0];
%       end
      
%       if nargin < 5
%         dims = 1:5;
%       end
      
%       if numel(x) ~= 5
%         error('Initial state does not have right dimension!');
%       end
      
      if ~iscolumn(x)
        x = x';
      end
      dims = 1:5;
      obj.dims = dims;
      
      obj.x = x;
      obj.xhist = obj.x;
      
%       obj.alphaMax = 0;
%       obj.aRange = [0; 0];
%       obj.dMax = [0; 0; 0; 0];
      
%       obj.pdim = 1:2;
%       obj.hdim = 3;
%       obj.vdim = 4;
      
      obj.nx = length(dims);
      obj.nu = 4;
      obj.nd = 4;

      obj.speed = u1;
      obj.ang_vel = u2;
      obj.acc = u3;
      obj.ang_acc = u4;
    end
  end % end methods
end % end classdef
