function dx = dynamics(obj, ~, x, u, d)
if iscell(x)
    dx{1} = obj.speed * sin(x{3});
    dx{2} = obj.speed * cos(x{3});
    dx{3} = obj.speed * obj.controller/obj.L;

    % Make sure we don't cross the grid limits
    cond = (abs(x{3}) <= (30*pi/180)) & (x{2} <= 240) & (x{2} >= 110);
%     dx{1} = dx{1} .* cond;
%     dx{2} = dx{2} .* cond;
%     dx{3} = dx{3} .* cond; 

else
  dx = zeros(obj.nx, 1);
  dx(1) = obj.speed * sin(x(3));
  dx(2) = obj.speed * cos(x(3));
  u = eval_u(obj.g, obj.controller, x');
  dx(3) = obj.speed * u/obj.L;
  cond = (abs(x(3)) <= (30*pi/180)) & (x(2) <= 240) & (x(2) >= 110);
%   dx(1) = dx(1) .* cond;
%   dx(2) = dx(2) .* cond;
%   dx(3) = dx(3) .* cond;
end

end