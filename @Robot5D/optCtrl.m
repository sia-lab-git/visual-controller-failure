function uOpt = optCtrl(obj, ~, ~, deriv, uMode)
% uOpt = optCtrl(obj, t, y, deriv, uMode)
%     Dynamics of the Plane5D
%         \dot{x}_1 = u_1 * cos(x_3) + d_1 (x position)
%         \dot{x}_2 = u_1 * sin(x_3) + d_2 (y position)
%         \dot{x}_3 = u_2                  (heading)
%         \dot{x}_4 = u_3 + d_3            (linear speed)
%         \dot{x}_5 = u_4 + d_4            (turn rate)

%% Input processing
if nargin < 5
  uMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

uOpt = cell(obj.nu, 1);
uOpt{1} = obj.speed;
uOpt{2} = obj.ang_vel;
uOpt{3} = obj.acc;
uOpt{4} = obj.ang_acc;
end
