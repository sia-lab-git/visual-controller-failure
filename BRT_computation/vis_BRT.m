clear all;
clf;

% specify the location of the .mat file with the solved value function
value_function = "true_BRT_gainCTE_0x74_gainHE_0x44.mat";

% specify the savefile name
filename = 'BRT.png';

% specify if plotting the ideal BRT
is_true = 1; % 1 for ideal BRT 0 for nn BRT

% specify the dtp
dtp = 180; % ignore if plotting the ideal BRT

%%
brt = load(value_function);

if is_true
    [grid_slice, data_slice] = proj(brt.g,brt.data(:,:,end),[0 0], 'min');
else
    % slice the nn BRT
    [grid_slice, data_slice] = proj(brt.g,brt.data(:,:,:,end),[0 1 0], dtp);
end

contourf(grid_slice.xs{1}, grid_slice.xs{2}, -data_slice(:,:,end), [0 0]);

axis square;
ylim([-20*pi/180 20*pi/180])

%convert y tics to degrees
ticks=get(gca,'ytick');
ticks=rad2deg(ticks);
ticks=arrayfun(@(x) sprintf('%d', round(x)), ticks, 'un', 0);
set(gca,'YTickLabels',ticks)
ylabel('HE(degrees)')
xlabel('CTE(m)')
saveas(gcf,filename)

