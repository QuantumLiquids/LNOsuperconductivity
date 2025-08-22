% PlotSingleOrbSpin1D.m
%
% Purpose
%   Plot on-site S^z vs x for single-orbital 1D ladder/cylinder.
%   Semilog-y for magnitude; supports real/complex JSON formats.
%
% Inputs (configured in-file)
%   Lx_phy, Ly_phy, t, J, Jperp, delta, Hole, D
%
% Data dependencies
%   Reads JSON: ../data/sz<... Pin>.json
%
% Behavior
%   Documentation only; plotting logic unchanged.
% Plot spin Sz configuration under pinning field
Lx_phy = 20;
Ly_phy = 3;
t = 3.0;
J = 1.0;
Jperp = 3.0;
delta = 0.5;
Hole = 0;
D = 5000;

Ly = 2 * Ly_phy; 
Lx = Lx_phy;

max_size = 600;  % Maximum circle size for hole density
max_arrow_length = 2; % Max arrow length for visual scaling

FileNamePostfix =[num2str(Ly_phy), 'x', num2str(Lx_phy), 't', num2str(t, '%.1f'), ...
            'J', num2str(J, '%.1f'), 'Jperp', num2str(Jperp, '%.1f'), 'delta', num2str(delta), ...
            'Hole', num2str(Hole),  ...
            'D', num2str(D), 'Pin.json'];
SpinData = jsondecode(fileread(['../data/sz', FileNamePostfix]));


% Extract x and y coordinates, charge density, and spin data
x_coor = zeros(1, size(SpinData, 1));
y_coor = zeros(1, size(SpinData, 1));
SpinDensity = zeros(1, numel(x_coor));

if isnumeric(SpinData)
    % real DMRG code
    for i = 1:numel(x_coor)
        x_coor(i) = fix(SpinData(i,1) / Ly) + 1;
        y_coor(i) = mod(SpinData(i,1), Ly) + 1;
        
        SpinDensity(i) = SpinData(i,2);
    end
else
    %complex DMRG code
    for i = 1:numel(x_coor)
        x_coor(i) = fix(SpinData{i}{1} / Ly) + 1;
        y_coor(i) = mod(SpinData{i}{1}, Ly) + 1;
        SpinDensity(i) = SpinData{i}{2}(1) + 1i * SpinData{i}{2}(2);
    end
end


% Flatten hole and spin density for scatter and arrow plot
start = 1;
spin_flat = (SpinDensity(start:end));
x_coor = x_coor(start:end);
y_coor = y_coor(start:end);

marker_colors{7} = [190, 184, 220] / 220;



semilogy(x_coor, abs(spin_flat), '-o'); hold on;

set(gca, 'FontSize', 24);
set(gca, 'LineWidth', 1.5);
set(get(gca, 'Children'), 'LineWidth', 2); % Set line width for all children (plots and errorbars)
xlabel('x', 'FontSize', 24);
ylabel('\langle S_i^z\rangle', 'FontSize', 24);
legend show;
box on;

hold off;

