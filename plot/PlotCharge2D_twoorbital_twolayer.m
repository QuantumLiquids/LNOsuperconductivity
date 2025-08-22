% PlotCharge2D_twoorbital_twolayer.m
%
% Purpose
%   Visualize on-site charge distribution for two-orbital, two-layer lattice.
%   Circle size and colormap encode charge density; OBC grid drawn for context.
%
% Inputs (configured in-file)
%   Lx_phy, Ly_phy, t1, t2, U, Jh, delta, Ele1, Ele2, D, orbital
%
% Data dependencies
%   Reads JSON: ../data/nf<LyxLx t1_t2 U Jh delta Ele1_Ele2 D Pin>.json
%
% Behavior
%   Documentation only; plotting logic unchanged.
% Plot charge distribution configuration
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

orbital = 'f'; % d for x^2-y^2, f for z^2
Ly = 2 * Ly_phy; % effective Ly
Lx = 2 * Lx_phy; % effective Lx

max_size = 300;  % Maximum circle size for charge density
max_arrow_length = 5; % Max arrow length for visual scaling

FileNamePostfix =[num2str(Ly_phy), 'x', num2str(Lx_phy), 't', num2str(t1), '_', num2str(t2), ...
            'U', num2str(U), 'Jh', num2str(Jh), 'delta', num2str(delta), ...
            'Ele', num2str(Ele1), '_',  num2str(Ele2), ...
            'D', num2str(D), 'Pin.json'];
ChargeData = jsondecode(fileread(['../data/nf', FileNamePostfix]));


% Extract x and y coordinates, charge density, and spin data
x_coor = zeros(1, size(ChargeData, 1));
y_coor = zeros(1, size(ChargeData, 1));
ChargeDensity = zeros(1, numel(x_coor));

% real DMRG code
for i = 1:numel(x_coor)
    site_idx = ChargeData(i,1);
    if(strcmp(orbital, 'd'))
        lx_offset = 0;
    else
        lx_offset = 1;
    end
    if(mod(fix(site_idx / Ly), 2)==lx_offset)
        x_coor(i) = fix(fix(site_idx / Ly)/2) + 1;
        y_coor(i) = mod(site_idx, Ly) + 1;
        ChargeDensity(i) = ChargeData(i,2);
    end
end


% Flatten charge density for scatter plot
start = 1;
charge_flat = (ChargeDensity(start:end));
x_coor = x_coor(start:end);
y_coor = y_coor(start:end);


x_coor = x_coor(charge_flat ~= 0);
y_coor = y_coor(charge_flat ~= 0);
charge_flat = charge_flat(charge_flat ~= 0);

% Plot the lattice points with open boundary conditions
figure;
hold on;

% Draw only the outer boundary lines
for row = 1:Ly
    plot([1, Lx/2], [row, row], 'k', 'LineWidth', 0.5); % Horizontal lines
end
for col = 1+lx_offset:1:Lx/2
    plot([col, col], [1, Ly], 'k', 'LineWidth', 0.5); % Vertical lines
end

% Plot charge density as colored circles
scatter(x_coor, y_coor, max_size * charge_flat / max(charge_flat), charge_flat, 'filled');
% charge_matrix = reshape(charge_flat, 2*Ly_phy, []);
% imagesc(1:Lx_phy, 1:2 * Ly_phy, charge_matrix);
colorbar;
my_colormap;
colormap(viridis);

% Set plot limits and axis properties
xlim([0.5, Lx/2 + 3]);
ylim([0.5, Ly + 0.5]);
axis equal;
set(gca, 'YDir', 'reverse'); % Ensure origin is at the bottom-left
axis off; % Remove axis ticks and labels
% axis equal;
% Add a manual legend
legend_x = Lx + 1;  % Position to the right of the plot
legend_y_start = Ly / 2 + 1;  % Start position at the top
legend_spacing = 0.5;  % Vertical spacing between legend items

% Charge density legend
digits = 1;   % Number of significant figures to keep
max_charge = max(charge_flat);
demonstrate_charge = round(max_charge, digits - floor(log10(abs(max_charge))) - 1);

% Example circle for maximum charge density
% scatter(legend_x, legend_y_start - legend_spacing, max_size * demonstrate_charge / max(charge_flat), demonstrate_charge, 'filled');
% text(legend_x + 0.3, legend_y_start - legend_spacing, ...
%     ['Charge = ', num2str(demonstrate_charge)], 'FontSize', 18);


hold off;