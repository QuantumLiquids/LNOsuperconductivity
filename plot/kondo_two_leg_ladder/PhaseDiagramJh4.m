% PhaseDiagramJh4.m
%
% Purpose
%   Plot phase diagram points and smoothed boundary for tilted two-leg with
%   J_H = 4, using scatter series and cubic-spline boundary curve.
%
% Behavior
%   Documentation only; plotting logic unchanged.
U = 8;
% Jh = 4, tilted 2-leg
colororder("gem");
C = colororder;

% Define marker size
my_marker_size = 100;
my_marker_size_square = 80;

% star, (pi/2, pi/2) state
U = [14,18,22,26,...
    22,26];
t_prime = [0.3 * ones(1, 4),...
    0.6 * ones(1,2)];
h1 = scatter(U, t_prime, 120, "filled", 'pentagram'); hold on;


% (0, pi) state
U = [2,2,2,6];
t_prime = [0.3,0.6,1,1];
h2 = scatter(U, t_prime, my_marker_size, "filled"); hold on;


%phase boundary line
%left - right line
x = [4,8, 20, 30];
y = [0,0.45 0.7, 0.8];
y_fine = linspace(min(y), max(y), 100);

% Perform cubic spline interpolation
x_fine = spline(y, x, y_fine);

% Plot the smooth curve
plot(x_fine, y_fine, 'k-');


set(gca, 'fontsize', 20);
set(gca, 'linewidth', 1.5);
set(get(gca, 'Children'), 'linewidth', 2);
xlabel('$U$', 'Interpreter', 'latex');
ylabel("$t'$", 'Interpreter', 'latex');
set(get(gca, 'XLabel'), 'FontSize', 20);
set(get(gca, 'YLabel'), 'FontSize', 20);
box on;