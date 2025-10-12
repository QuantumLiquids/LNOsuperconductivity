% plot_kondo_two_leg/plot_spin_corr.m
%
% Purpose
%   Visualize equal-time spin correlations on the two-leg tilted Kondo lattice.
%   (Single-layer model)
%   Marker size encodes |corr| and color encodes sign; a legend is drawn.
%
% Inputs (configured in-file)
%   L, t2, Jk, U, Db  - model and truncation parameters used to form the
%                       data file postfix.
% 
% Data dependencies
%   Reads JSON from ../../data/ with names:
%     szsz<t2...Jk...U...Lx...D...>.json
%     spsm<t2...Jk...U...Lx...D...>.json
%     smsp<t2...Jk...U...Lx...D...>.json
%
% Data files miss D information have been appendixed as D=0.
% 
% Other dependencies
%   Requires KondoTilted2LegLattice.m in the same folder for geometry.
%
% Behavior
%   This header adds documentation only; plotting logic is unchanged.
clear;
close all;

% Parameters
L = 20;         % Number of unit cells
t2 =0.3;
Jk = -4;
U = 2;
Db = 0;
base_marker_size = 300;  % Max size for largest magnitude


% % light purple and red
positive_spin_color = [142   139	254]/256;
negative_spin_color = [232   132	130]/256;
% light orange and blue
% positive_spin_color = [250	168	53	]/256;
% negative_spin_color = [104	204	217		]/256;
% Load spin correlation data
FileNamePostfix = ['t2', num2str(t2), 'Jk', num2str(Jk), 'U', num2str(U), 'Lx', num2str(L),  'D', num2str(Db), '.json'];
% FileNamePostfix = ['t2', num2str(t2), 'Jk', num2str(Jk), 'U', num2str(U),'.json'];
SpinCorrDataZZ = jsondecode(fileread(['../../data/szsz', FileNamePostfix]));
SpinCorrDataPM = jsondecode(fileread(['../../data/spsm', FileNamePostfix]));
SpinCorrDataMP = jsondecode(fileread(['../../data/smsp', FileNamePostfix]));

% Extract reference site and correlation values
data_num = numel(SpinCorrDataZZ);
ref_site_idx = SpinCorrDataZZ{1}{1}(1);  % Reference site (0-indexed)
target_site_idx = zeros(1, data_num);
SpinCorr = zeros(1, data_num);

for i = 1:data_num
    target_site_idx(i) = SpinCorrDataZZ{i}{1}(2);
    SpinCorr(i) = SpinCorrDataZZ{i}{2} + SpinCorrDataPM{i}{2};
end

% Prepare correlation data [site_index, correlation_value]
corr_data = [target_site_idx', SpinCorr'];

% Create lattice and draw bonds
lattice = KondoTilted2LegLattice(4*L, 'OBC');
figure;
lattice.drawLattice(1.5, 0);
hold on;

% 1. Filter for extended sites
is_extended = arrayfun(@(idx) lattice.isExtendedSite(idx), corr_data(:,1));
corr_data_extended = corr_data(is_extended, :);

% 2. Separate reference site and other extended sites
is_ref = (corr_data_extended(:,1) == ref_site_idx);
ref_corr = corr_data_extended(is_ref, :);
other_corr = corr_data_extended(~is_ref, :);

% Flaten data
% other_corr(:,2) = sign(other_corr(:,2)).*sqrt(abs(other_corr(:,2)));

% 3. Get coordinates for reference site
[x_ref, y_ref] = lattice.indexToCoord(ref_site_idx);

% 4. Compute marker sizes for other sites (scale magnitude)
max_magnitude = max(abs(other_corr(:,2)));
if max_magnitude == 0
    max_magnitude = 1;  % Avoid division by zero
end

marker_sizes = base_marker_size * abs(other_corr(:,2)) / max_magnitude;

% 5. Plot correlations for non-reference extended sites
for i = 1:size(other_corr, 1)
    site_idx = other_corr(i, 1);
    corr_val = other_corr(i, 2);
    [x, y] = lattice.indexToCoord(site_idx);

    % Choose color based on sign
    if corr_val >= 0
        color = positive_spin_color;
    else
        color = negative_spin_color;
    end

    % Plot filled circle with size proportional to magnitude
    scatter(x, y, marker_sizes(i), color, 'filled', 'MarkerEdgeColor', 'k');
end

% 6. Highlight reference site with a star
plot(x_ref, y_ref, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
% 7. Add bubble legend like the sketch (circles with numbers below)
%    - Use one-significant-digit magnitude of |data|max as the legend scale
%    - Size mapping identical to main plot: size ~ |value| / |data|max
real_max_abs = max(abs(other_corr(:,2)));
if real_max_abs == 0
    real_max_abs = 1; % avoid division by zero, degenerate case
end

% One significant digit magnitude for legend values
legend_max_abs = str2double(sprintf('%.1g', real_max_abs));
if legend_max_abs == 0
    legend_max_abs = 1; % fallback
end

% Symmetric legend values (negative: filled, positive: hollow)
legend_values = legend_max_abs * [-1, -0.5, -0.1, 0.1, 0.5, 1];

% Layout along the right side, horizontally
ax = gca;
x_lim = ax.XLim;
y_lim = ax.YLim;
plot_width = x_lim(2) - x_lim(1);
plot_height = y_lim(2) - y_lim(1);

legend_y = y_lim(1) + plot_height * 0.12;             % baseline for circles
text_y   = legend_y - plot_height * 0.06;             % text slightly below
legend_x0 = x_lim(1) + 0.5 *plot_width;             % closer to the plot (move left)
dx = plot_width * 0.08;                                % horizontal spacing

for k = 1:numel(legend_values)
    vx = legend_x0 + (k-1) * dx;
    vv = legend_values(k);
    sz = base_marker_size * abs(vv) / real_max_abs;
    if sz <= 0
        sz = 1;
    end

    if vv < 0
        % negative: filled circle
        scatter(vx, legend_y, sz, negative_spin_color, 'filled', 'MarkerEdgeColor', 'k');
    else
        % positive: filled with positive color (match main plot)
        scatter(vx, legend_y, sz, positive_spin_color, 'filled', 'MarkerEdgeColor', 'k');
    end

    % number below each circle, using one significant digit
    text(vx, text_y, sprintf('%.1g', vv), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', 'FontSize', 14, 'FontName', 'Arial');
end

% title(sprintf('Spin Correlations (Reference Site: %d)', ref_site_idx));

% 8. Add model parameters text box
param_str = sprintf('t'' = %.1ft\nJ_H = %.1ft\nU = %.1ft', t2, -Jk, U);
annotation('textbox', [0.2, 0.82, 0.15, 0.1], 'String', param_str, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'k', ...
    'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold', 'Margin', 5);

hold off;
axis off;
axis equal;

% -----------------------------------------------------------------------------
% Transparent vector export to figures/
try
    set(gcf, 'Color','none', 'InvertHardcopy','off', 'Renderer','painters');
    set(findall(gcf, 'Type','axes'), 'Color','none');
    this_file = mfilename('fullpath');
    if isempty(this_file)
        this_dir = pwd;
    else
        this_dir = fileparts(this_file);
    end
    fig_dir = fullfile(this_dir, 'figures');
    if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

    % Build filename for single-parameter figure
    jh = -Jk; % J_H = -Jk per annotation
    name_tokens = { kv_token('jh', jh, false), ...
                    kv_token('t2', t2, true), ...
                    kv_token('u',  U,  false), ...
                    kv_token('lx', L,  false) };
    base_name = ['kondo_2leg_spin_corr_', strjoin(name_tokens, '_')];
    pdf_path = fullfile(fig_dir, [base_name, '.pdf']);
    eps_path = fullfile(fig_dir, [base_name, '.eps']);

    exportgraphics(gcf, pdf_path, 'ContentType','vector', 'BackgroundColor','none');
    print(gcf, '-depsc', '-painters', '-r600', eps_path);
catch ME
    warning(ME.identifier, '%s', ME.message);
end

% Local helpers for naming
function tok = kv_token(key, val, always_hyphen)
    s = fmt_num_short(val);
    needs_hyphen = always_hyphen || contains(s,'.') || contains(s,'-');
    if needs_hyphen
        tok = [key, '-', s];
    else
        tok = [key, s];
    end
end

function s = fmt_num_short(x)
    s = sprintf('%.15g', x);
end