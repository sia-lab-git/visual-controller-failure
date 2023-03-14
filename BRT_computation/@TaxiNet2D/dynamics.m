function dx = dynamics(obj, ~, x, u, d)
    if iscell(x)
        dx{1} = obj.speed * sin(x{2});
        u = obj.gainCTE * x{1} + obj.gainHE * x{2} * 180/pi; %Phi in degrees
        u = u * pi/180; %Phi in radians
        dx{2} = obj.speed * tan(u)/obj.L;
    
        % Make sure we don't cross the angle of [-30, 30] degrees
        cond = (abs(x{2}) <= (30*pi/180));
        dx{1} = dx{1} .* cond;
        dx{2} = dx{2} .* cond;
    else
        dx = zeros(obj.nx, 1);
        dx(1) = obj.speed * sin(x(2));
        u = obj.gainCTE * x(1) + obj.gainHE * x(2) * 180/pi;
        u = u * pi/180;
        dx(2) = obj.speed * tan(u)/obj.L;
        cond = (abs(x(2)) <= (30*pi/180));
        dx(1)= dx(1).* cond;
        dx(2) = dx(2) .* cond;
    end
end