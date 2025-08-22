% PlotSpinCorr1D.m
%
% Purpose
%   Plot |S_i·S_j| vs distance (1D) without pinning for two-orbital model.
%   Semilog-y magnitude based on combined sz0sz, sp0sm, sm0sp.
%
% Inputs (configured in-file)
%   Lx_phy, Ly_phy, t1, t2, U, Jh, delta, Ele1, Ele2, D
%
% Data dependencies
%   Reads JSON: ../data/sz0sz, sp0sm, sm0sp with <... NoPin>.json postfix
%
% Behavior
%   Documentation only; plotting logic unchanged.
% Plot spin S*S corrrelation without pinning field
Lx_phy = 16;
Ly_phy = 2;
t1 = 0.635;
t2 = 0.483;
U = 5.796;
Jh = 0.2415;
delta = 0;
Ele1 = 48;
Ele2 = 48;
D = 4000;

Ly = 2 * Ly_phy; 
Lx = 2 * Lx_phy;

max_size = 600;  % Maximum circle size for hole density
max_arrow_length = 1; % Max arrow length for visual scaling

FileNamePostfix =[num2str(Ly_phy), 'x', num2str(Lx_phy), 't', num2str(t1), '_', num2str(t2), ...
            'U', num2str(U), 'Jh', num2str(Jh), 'delta', num2str(delta), ...
            'Ele', num2str(Ele1), '_',  num2str(Ele2), ...
            'D', num2str(D), 'Pin.json'];
SpinDataz = jsondecode(fileread(['../data/sz0sz', FileNamePostfix]));
SpinData1 = jsondecode(fileread(['../data/sp0sm', FileNamePostfix]));
SpinData2 = jsondecode(fileread(['../data/sm0sp', FileNamePostfix]));

% Extract x and y coordinates, charge density, and spin data
x_coor = zeros(1, size(SpinDataz, 1));
y_coor = zeros(1, size(SpinDataz, 1));
SpinDensity = zeros(1, numel(x_coor));

% real DMRG code
refer_site = SpinDataz{1}{1}(1);

% x_coor(1) = 0;
% y_coor(1) = 0;
% SpinDensity(1)  = 0;  
for i = 1:numel(x_coor)
    % if(mod(fix(SpinDataz{i}{1}(2) / Ly), 2)==1)
        x_coor(i) = fix(SpinDataz{i}{1}(2) / Ly) + 1;
        y_coor(i) = mod(SpinDataz{i}{1}(2), Ly) + 1;
        SpinDensity(i) = SpinDataz{i}{2} + 0.5*(SpinData1{i}{2} + SpinData2{i}{2});
    % end
end

% Flatten hole and spin density for scatter and arrow plot
start = 1;
spin_flat = (SpinDensity(start:end));
x_coor = x_coor(start:end);
y_coor = y_coor(start:end);

x_phy = fix(x_coor /2);

% Plot the lattice points with open boundary conditions
% figure;
% hold on;

marker_colors{7} = [190, 184, 220] / 220;

loglog(x_coor, abs(spin_flat), '-o'); hold on;

set(gca, 'FontSize', 24);
set(gca, 'LineWidth', 1.5);
set(get(gca, 'Children'), 'LineWidth', 2); % Set line width for all children (plots and errorbars)
xlabel('x', 'FontSize', 24);
ylabel('\langle S_i \cdot S_j\rangle', 'FontSize', 24);
legend show;
box on;

hold off;