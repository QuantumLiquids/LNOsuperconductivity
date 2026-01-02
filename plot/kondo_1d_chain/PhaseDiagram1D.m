% PhaseDiagram1D.m
%
% Purpose
%   Plot phase diagram points and a smoothed phase boundary for 1D case,
%   with annotated scatter series and cubic-spline boundary curve.
%
% Behavior
%   Documentation only; plotting logic unchanged.

% Use palette aligned with paper-ready scheme (Option A)
fm_color  = [27 158 119]/255;   % deep teal  → FM
sdw_color = [117 112 179]/255;  % purple     → 2 k_F-SDW

% Define marker size
my_marker_size = 100;
my_marker_size_square = 80;

% triangle, 2k_F-SDW state (was star)
U = [(0:2:12),...
    0,2,4,6,...
    0,2,...
    0];
Jh = [0 * ones(1,7),...
    2 * ones(1, 4),...
    4 * ones(1,2),...
    6 * ones(1,1)];
U_tri = U; Jh_tri = Jh;
h1 = scatter(U, Jh, 120, "filled", '^', 'MarkerFaceColor', fm_color, 'MarkerEdgeColor', 'none', 'HandleVisibility','off'); hold on;


% (0, pi) state
U = [10,12,...
    8,10,12,...
    (6:2:12),...
    4:2:12,...
    0,...
    (0:2:12)];
Jh = [4 * ones(1,2),...
    6 * ones(1,3),...
    8*ones(1,4),...
    10*ones(1,5), ...
    13,...
    15*ones(1,7)];
U_opi = U; Jh_opi = Jh;
h2 = scatter(U, Jh, my_marker_size, "filled", 'MarkerFaceColor', sdw_color, 'MarkerEdgeColor', 'none', 'HandleVisibility','off'); hold on;

%phase boundary line
%left - right line
x = [0,   4, 8, 10, 12];
y = [12.3,9, 5, 3.5, 2.5]-2;
y_fine = linspace(min(y), max(y), 100);

% Perform cubic spline interpolation (y-range extended to cover all scatter)
y_all = [y, Jh_tri, Jh_opi];
y_fine = linspace(min(y_all), max(y_all), 200);
x_fine = spline(y, x, y_fine);

% Establish axes limits from data (fix x to [0,12])
x_max = 12;  % requested boundary
y_min = min([Jh_tri, Jh_opi, y_fine]) - 0.2;
y_max = max([Jh_tri, Jh_opi, y_fine]) + 0.2;
% Background phase fills using same hue family: data points = full color,
% background = same hue but lighter + transparent.
% Auto-detect which side of boundary is SDW/FM using median U of each set.
% Boundary reference x at mid-height:
y0 = median(y_fine);
x_boundary_mid = spline(y, x, y0);
sdw_left = median(U_opi) <= x_boundary_mid; % SDW cluster on the left side?

mix = 0.5;              % 0->original, 1->white (lighter background)
face_alpha = 0.25;      % transparency of background fill
sdw_bg = (1-mix)*sdw_color + mix*[1 1 1];
fm_bg  = (1-mix)*fm_color  + mix*[1 1 1];
if sdw_left
    left_fill_color  = sdw_bg;
    right_fill_color = fm_bg;
else
    left_fill_color  = fm_bg;
    right_fill_color = sdw_bg;
end

% Left region polygon (x from 0 to boundary) — ensure full vertical coverage
x_left = [zeros(size(y_fine)), fliplr(x_fine)];
y_left = [y_fine,            fliplr(y_fine)];
patch(x_left, y_left, left_fill_color, 'EdgeColor', 'none', 'FaceAlpha', face_alpha); hold on;

% Right region polygon (boundary to x_max) — fill to the right border
x_right = [x_fine, x_max*ones(size(y_fine(end:-1:1)))];
y_right = [y_fine, y_fine(end:-1:1)];
patch(x_right, y_right, right_fill_color, 'EdgeColor', 'none', 'FaceAlpha', face_alpha); hold on;

% Plot the smooth boundary curve on top
plot(x_fine, y_fine, 'k-', 'LineWidth', 2); hold on;

% Phase labels positioned near the mid-height (reuse y0/x_boundary_mid)
text(1.5*x_boundary_mid, y0/2, '2k_F-SDW', 'FontName','Arial', 'FontSize', 20, ...
    'FontWeight','bold', 'Color', sdw_color, 'HorizontalAlignment','center');
text((x_boundary_mid + x_max)/2, y0*3/2, 'FM', 'FontName','Arial', 'FontSize', 20, ...
    'FontWeight','bold', 'Color', fm_color, 'HorizontalAlignment','center');

% 
% %PDW - gap line
% x_start = spline(y, x, 0.015);
% plot([x_start,8],[0.015, 0.015],'k-' );
% %s-wave -CDW line
% 
% x_target = 1.3;
% y_target = interp1(x_fine, y_fine, x_target, 'linear');
% plot([0, x_target],[0.015, y_target],'k-' );

set(gca, 'fontsize', 20);
set(gca, 'linewidth', 1.5);
set(get(gca, 'Children'), 'linewidth', 2);
xlim([0 x_max]); 
ylim([0 15]);
xlabel('U', 'FontName','Arial');
ylabel("J_H", 'FontName','Arial');
set(get(gca, 'XLabel'), 'FontSize', 20);
set(get(gca, 'YLabel'), 'FontSize', 20);
box on;