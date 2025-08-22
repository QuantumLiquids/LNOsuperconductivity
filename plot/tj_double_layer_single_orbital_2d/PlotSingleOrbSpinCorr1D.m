% PlotSingleOrbSpinCorr1D.m
%
% Purpose
%   Plot |S_i·S_j| vs distance for multiple bond dimensions D (1D profile).
%   Semilog-y magnitude; legend indicates D.
%
% Inputs (configured in-file)
%   Lx_phy, Ly_phy, t, J, Jperp, delta, Hole, D_list
%
% Data dependencies
%   Reads JSON: ../data/sz0sz, sp0sm, sm0sp with <... NoPin>.json postfix
%
% Behavior
%   Documentation only; plotting logic unchanged.
% Define parameters
Lx_phy = 20;
Ly_phy = 2;
t = 3.0;
J = 1.0;
Jperp = 3.0;
delta = 0.3;
Hole = 0;

Ly = 2 * Ly_phy; 
Lx = Lx_phy;

start = 8;  % Starting index for plotting

% Define list of D values
D_list = [3000, 8000];

% Read SpinDataz for the first D to get x_coor and y_coor
D = D_list(1);
FileNamePostfix = [num2str(Ly_phy), 'x', num2str(Lx_phy), 't', num2str(t, '%.1f'), ...
    'J', num2str(J, '%.1f'), 'Jperp', num2str(Jperp, '%.1f'), 'delta', num2str(delta), ...
    'Hole', num2str(Hole), 'D', num2str(D), 'NoPin.json'];
SpinDataz = jsondecode(fileread(['../../data/sz0sz', FileNamePostfix]));

% Extract x_coor and y_coor
if iscell(SpinDataz)  % Real DMRG code
    num_sites = numel(SpinDataz);
    x_coor = zeros(1, num_sites);
    y_coor = zeros(1, num_sites);
    for i = 1:num_sites
        x_coor(i) = fix(SpinDataz{i}{1}(2) / Ly) + 1;
        y_coor(i) = mod(SpinDataz{i}{1}(2), Ly) + 1;
    end
else  % Complex DMRG code
    x_coor = fix(SpinDataz(:,3) / Ly) + 1;
    x_coor = x_coor';
    y_coor = mod(SpinDataz(:,3), Ly) + 1;
    y_coor = y_coor';
end

% Truncate x_coor for plotting
x_coor_plot = x_coor(start:end);

% Create figure and hold on for multiple plots
figure;

% Loop over each D in D_list
for D = D_list
    FileNamePostfix = [num2str(Ly_phy), 'x', num2str(Lx_phy), 't', num2str(t, '%.1f'), ...
        'J', num2str(J, '%.1f'), 'Jperp', num2str(Jperp, '%.1f'), 'delta', num2str(delta), ...
        'Hole', num2str(Hole), 'D', num2str(D), 'NoPin.json'];
    
    SpinDataz = jsondecode(fileread(['../../data/sz0sz', FileNamePostfix]));
    SpinData1 = jsondecode(fileread(['../../data/sp0sm', FileNamePostfix]));
    SpinData2 = jsondecode(fileread(['../../data/sm0sp', FileNamePostfix]));
    
    % Compute SpinDensity
    if iscell(SpinDataz)  % Real DMRG code
        SpinDensity = zeros(1, numel(x_coor));
        for i = 1:numel(x_coor)
            SpinDensity(i) = SpinDataz{i}{2} + 0.5*(SpinData1{i}{2} + SpinData2{i}{2});
        end
    else  % Complex DMRG code
        SpinDensity = SpinDataz(:,2)' + 0.5*(SpinData1(:,2)' + SpinData2(:,2)');
    end
    
    % Truncate SpinDensity for plotting
    spin_flat = SpinDensity(start:end);
    
    % Plot with label for legend
    semilogy(x_coor_plot, abs(spin_flat), '-o', 'DisplayName', ['D=', num2str(D)]);hold on;
end

% Set axes properties
set(gca, 'FontSize', 24);
set(gca, 'LineWidth', 1.5);
set(get(gca, 'Children'), 'LineWidth', 2);
xlabel('x', 'FontSize', 24);
ylabel('\langle S_i \cdot S_j\rangle', 'FontSize', 24);
legend('show');  % Display legend with D values
box on;

hold off;