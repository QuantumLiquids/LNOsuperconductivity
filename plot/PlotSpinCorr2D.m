% Plot spin S*S corrrelation without pinning field
Lx_phy = 16;
Ly_phy = 2;
t1 = 0.483;
t2 = 0.635;
U = 5.796;
Jh = 0.483;
delta = 0.3;
Ele1 = 48;
Ele2 = 48;
D = 2000;

Ly = 2 * Ly_phy; 
Lx = 2 * Lx_phy;
max_size = 600;  % Maximum circle size for hole density
max_arrow_length = 100; % Max arrow length for visual scaling

FileNamePostfix =[num2str(Ly_phy), 'x', num2str(Lx_phy), 't', num2str(t1), '_', num2str(t2), ...
            'U', num2str(U), 'Jh', num2str(Jh), 'delta', num2str(delta), ...
            'Ele', num2str(Ele1), '_',  num2str(Ele2), ...
            'D', num2str(D), 'NoPin.json'];
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
    if(mod(fix(SpinDataz{i}{1}(2) / Ly), 2)==0)
        x_coor(i) = fix(SpinDataz{i}{1}(2) / Ly) + 1;
        y_coor(i) = mod(SpinDataz{i}{1}(2), Ly) + 1;
        SpinDensity(i) = SpinDataz{i}{2} + 0.5*(SpinData1{i}{2} + SpinData2{i}{2});
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
xlim([0.5, Lx + 3]);
ylim([0.5, Ly + 0.5]);
axis equal;
set(gca, 'YDir', 'reverse'); % Ensure origin is at the bottom-left
axis off; % Remove axis ticks and labels

% Add a manual legend
legend_x = Lx + 1;  % Position to the right of the plot
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