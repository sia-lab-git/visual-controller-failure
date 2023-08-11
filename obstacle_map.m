function data = obstacle_map(grid, distance_map, offset, map_bounds)
    data = zeros(grid.shape);
    [l,w] = size(distance_map);
    bound_x = map_bounds(2);
    bound_y =  map_bounds(1);
    for i=1:length(grid.vs{1})
        for j=1:length(grid.vs{2})
            for k=1:length(grid.vs{3})
                for ll=1:length(grid.vs{4})
                    for m=1:length(grid.vs{5})
                    pos_x = grid.vs{1}(i) + offset(1);
                    pos_y = grid.vs{2}(j) + offset(2);
                    heading = grid.vs{3}(k);
                    speed = grid.vs{4}(ll);
                    ang_speed = grid.vs{4}(m);
                    % convert the pos into pixel to query map
                    pixel_x = floor(pos_x*w/bound_x);
                    pixel_y = floor(pos_y*l/bound_y);
                    data(i,j,k,ll,m) = distance_map(pixel_y,pixel_x);
                end
            end
        end
        end
    end
end

% 
% function data = obstacle_map(grid, distance_map, offset, map_bounds)
%     data = zeros(grid.shape);
%     [l,w] = size(distance_map);
%     bound_x = map_bounds(2);
%     bound_y =  map_bounds(1);
%     for i=1:length(grid.vs{1})
%         for j=1:length(grid.vs{2})
%             for k=1:length(grid.vs{3})
%                 pos_x = grid.vs{1}(i) + offset(1);
%                 pos_y = grid.vs{2}(j) + offset(2);
%                 heading = grid.vs{3}(k);
% %                         speed = grid.vs{4}(l);
% %                         ang_speed = grid.vs{4}(m);
%                 % convert the pos into pixel to query map
%                 pixel_x = floor(pos_x*w/bound_x);
%                 pixel_y = floor(pos_y*l/bound_y);
%                 data(i,j,k) = distance_map(pixel_y,pixel_x);
%             end
%          end
%      end
% end


                
                
                
