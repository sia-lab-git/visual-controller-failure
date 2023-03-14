classdef TaxiNet2D < DynSys
  properties
    % Speed
    speed
    
    % Length
    L

    % Gain for crosstrack error
    gainCTE

    % Gain for heading
    gainHE

    dims
  end
  
  methods
      function obj = TaxiNet2D(x, gainCTE, gainHE, speed, L)

      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      if ~iscolumn(x)
        x = x';
      end

      if nargin < 4
        speed = 5.0;
      end

      if nargin < 5
        L = 5.0;
      end
      
      % Default number of dims if not provided
      dims = 1:2;

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
    end
    
  end % end methods
end % end classdef
