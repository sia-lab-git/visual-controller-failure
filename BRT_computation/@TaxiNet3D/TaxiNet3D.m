classdef TaxiNet3D < DynSys
  properties
    % Speed
    speed
    
    % Length
    L

    % Gain for crosstrack error
    gainCTE

    % Gain for heading
    gainHE

    % Predicted controller
    controller

    % Grid for teh controller
    g

    dims
  end
  
  methods
      function obj = TaxiNet3D(x, gainCTE, gainHE, controller, speed, L, g)

      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      if ~iscolumn(x)
        x = x';
      end

      if nargin < 5
        speed = 5.0;
      end

      if nargin < 6
        L = 5.0;
      end
      
      if nargin < 7
        g = nan;
      end

      % Default number of dims if not provided
      dims = 1:3;

      % Basic vehicle properties
      obj.dims = dims;
      obj.nx = length(dims);

      obj.nu = 1;
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.gainCTE = gainCTE;
      obj.gainHE = gainHE;
      obj.speed = speed;
      obj.L = L;
      obj.g = g;
      obj.controller = controller;
    end
    
  end % end methods
end % end classdef
