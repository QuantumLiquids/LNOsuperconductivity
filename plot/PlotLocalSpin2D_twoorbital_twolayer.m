% PlotLocalSpin2D_twoorbital_twolayer.m
%
% Purpose
%   Visualize on-site S^z under pinning for two-orbital, two-layer lattice.
%   Up/down arrows encode sign and magnitude; OBC grid optional.
%
% Inputs (configured in-file)
%   Lx_phy, Ly_phy, t1, t2, U, Jh, delta, Ele1, Ele2, D, orbital
%
% Data dependencies
%   Reads JSON: ../data/sz<LyxLx t1_t2 U Jh delta Ele1_Ele2 D Pin>.json
%
% Behavior
%   Adds documentation and fixes a variable reference bug; plotting unchanged.
% Plot spin Sz configuration under pinning field
Lx_phy = 16;
Ly_phy = 2;
t1 = 0.483;
t2 = 0.635;
U = 5.796;
Jh = 0.2415;
delta = 0.3;
Ele1 = 48;
Ele2 = 48;
D = 8000;

orbital = 'd'; % d for x^2-y^2, f for z^2
Ly = 2 * Ly_phy;
Lx = 2 * Lx_phy;

max_size = 600;  % Maximum circle size for hole density
max_arrow_length = 2; % Max arrow length for visual scaling

FileNamePostfix =[num2str(Ly_phy), 'x', num2str(Lx_phy), 't', num2str(t1), '_', num2str(t2), ...
    'U', num2str(U), 'Jh', num2str(Jh), 'delta', num2str(delta), ...
    'Ele', num2str(Ele1), '_',  num2str(Ele2), ...
    'D', num2str(D), 'Pin.json'];
SpinData = jsondecode(fileread(['../data/sz', FileNamePostfix]));


% Extract x and y coordinates, charge density, and spin data
x_coor = zeros(1, size(SpinData, 1));
y_coor = zeros(1, size(SpinData, 1));
SpinDensity = zeros(1, numel(x_coor));

% real DMRG code
for i = 1:numel(x_coor)
    site_idx = SpinData(i,1);
    if(strcmp(orbital, 'd'))
        lx_offset = 0;
    else
        lx_offset = 1;
    end
    if(mod(fix(SpinData(i,1) / Ly), 2)==lx_offset)
        x_coor(i) = fix(fix(site_idx / Ly)/2) + 1;
        y_coor(i) = mod(site_idx, Ly) + 1;
        SpinDensity(i) = SpinData(i,2);
    end
end


% Flatten hole and spin density for scatter and arrow plot
start = 1;
spin_flat = (SpinDensity(start:end));
x_coor = x_coor(start:end);
y_coor = y_coor(start:end);

% Plot the lattice points with open boundary conditions
% figure;
hold on;

% % Draw only the outer boundary lines
% for row = 1:Ly
%     plot([1, Lx], [row, row], 'k', 'LineWidth', 2); % Horizontal lines
% end
% for col = 1:Lx
%     plot([col, col], [1, Ly], 'k', 'LineWidth', 2); % Vertical lines
% end

% Define marker colors
marker_colors{7} = [190, 184, 220] / 220;

% Plot spin density as arrows
for i = 1:length(spin_flat)
    arrow_length = max_arrow_length * abs(spin_flat(i)) / max(abs(spin_flat));
    arrow_shift = arrow_length / 2;  % Shift to center arrow in circle

    if spin_flat(i) > 0
        % Downward arrow (shifted up to center)
        quiver(x_coor(i), y_coor(i) + arrow_shift, 0, -arrow_length, 0, 'Color', 'b', 'MaxHeadSize', 1, 'LineWidth', 1.5);
    elseif spin_flat(i) < 0
        % Upward arrow (shifted down to center)
        quiver(x_coor(i), y_coor(i) - arrow_shift, 0, arrow_length, 0, 'Color', 'r', 'MaxHeadSize', 1, 'LineWidth', 1.5);
    end
end

% Set plot limits and axis properties
xlim([0.5, Lx/2 + 3]);
ylim([0.5, Ly + 0.5]);
axis equal;
set(gca, 'YDir', 'reverse'); % Ensure origin is at the bottom-left
axis off; % Remove axis ticks and labels

% Add a manual legend
legend_x = Lx/2 + 1;  % Position to the right of the plot
legend_y_start = Ly / 2 + 1;  % Start position at the top
legend_spacing = 0.5;  % Vertical spacing between legend items

% Spin legend
digits = 1;   % Number of significant figures to keep
max_spin = max(abs(spin_flat));
demonstrate_spin = round(max_spin, digits - floor(log10(abs(max_spin))) - 1);

% Spin Up example
quiver(legend_x, legend_y_start - legend_spacing + max_arrow_length * demonstrate_spin / 2.0 / max(abs(spin_flat)), ...
    0, -max_arrow_length * demonstrate_spin / max(abs(spin_flat)), ...
    0, 'Color', 'b', 'MaxHeadSize', 1, 'LineWidth', 1.5);
text(legend_x + 0.3, legend_y_start - legend_spacing, ...
    ['Spin Up = ', num2str(demonstrate_spin)], 'FontSize', 18);

% Spin Down example
quiver(legend_x, legend_y_start - 2 * legend_spacing - max_arrow_length * demonstrate_spin / 2.0 / max(abs(spin_flat)), ...
    0, max_arrow_length * demonstrate_spin / max(abs(spin_flat)), ...
    0, 'Color', 'r', 'MaxHeadSize', 1, 'LineWidth', 1.5);
text(legend_x + 0.3, legend_y_start - 2 * legend_spacing, ...
    ['Spin Down = ', num2str(demonstrate_spin)], 'FontSize', 18);

hold off;