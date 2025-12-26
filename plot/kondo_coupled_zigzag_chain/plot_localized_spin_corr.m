% plot_kondo_ladder/plot_itinerate_spin_corr.m
%
% Purpose
%   Visualize equal-time spin correlations on the tilted zig-zag Kondo lattice
%   for arbitrary Ly (number of zig-zag chains). Marker size encodes |corr|
%   and color encodes the sign; a bubble legend is drawn.
%
% Inputs (configured in-file)
%   Lx, Ly, t2, Jk, U, Db  - model/truncation params used to form the data file
%                            postfix. Legacy Ly=2 data omit the Ly token.
%
% Data dependencies
%   Reads JSON from ../../data/ with names like:
%     szszt2...Jk...U...Ly...Lx...D...json
%   The helper `resolve_corr_path` falls back to Ly-less filenames for Ly=2.
%
% Other dependencies
%   Requires TiltedZigZagLattice.m in the same folder for geometry.
clear;
close all;

% Parameters
% Lx counts unit cells along the zig-zag chain direction; each unit cell hosts Ly
% itinerant orbitals in this effective 2D mapping.
Lx = 20;
Ly = 4;         % New generalized ladder width
 t2 = 0.3;
Jk = -4;
U  = 2;
Db = 10000;
base_marker_size = 300;  % Max size for largest magnitude
transparent_background = false;  % Set true to export with transparent background

% Colors (light purple and red)
positive_spin_color = [142   139  254]/256;
negative_spin_color = [232   132  130]/256;

% Load spin correlation data with Ly-aware postfix handling
postfix = build_postfix(t2, Jk, U, Ly, Lx, Db);
base_path = '../../data/';
SpinCorrDataZZ = jsondecode(fileread(resolve_corr_path(base_path, 'szsz', postfix, Ly)));
SpinCorrDataPM = jsondecode(fileread(resolve_corr_path(base_path, 'spsm', postfix, Ly)));
SpinCorrDataMP = jsondecode(fileread(resolve_corr_path(base_path, 'smsp', postfix, Ly)));

data_num = numel(SpinCorrDataZZ);
ref_site_raw = SpinCorrDataZZ{1}{1}(1);
target_site_idx = zeros(1, data_num);
SpinCorr = zeros(1, data_num);
for i = 1:data_num
    target_site_idx(i) = SpinCorrDataZZ{i}{1}(2);
    SpinCorr(i) = SpinCorrDataZZ{i}{2} + SpinCorrDataPM{i}{2};
end

raw_indices = [ref_site_raw; target_site_idx'];
% Legacy data stored even/odd indices for itinerant/localized sites. Modern
% files already use itinerant-only indexing. Accept both without breaking.
if any(mod(raw_indices, 2) ~= 0)
    error('Spin correlation dataset includes localized-site indices (odd). Expected even indices for itinerant electrons.');
end

ref_site_idx = ref_site_raw / 2;
site_indices = target_site_idx' / 2;
corr_data = [site_indices, SpinCorr'];

lattice = TiltedZigZagLattice(Ly, Lx, 'OBC');
figure;
lattice.drawLattice(1.5, 0);
hold on;

is_valid = all(corr_data(:,1) >= 0 & corr_data(:,1) < lattice.N, 2);
if any(~is_valid)
    error('Correlation targets outside lattice range detected.');
end

other_corr = corr_data(corr_data(:,1) ~= ref_site_idx, :);

[x_ref, y_ref] = lattice.indexToCoord(ref_site_idx);

if isempty(other_corr)
    max_magnitude = 0;
else
    max_magnitude = max(abs(other_corr(:,2)));
end
if max_magnitude == 0
    max_magnitude = 1;
end

marker_sizes = base_marker_size * abs(other_corr(:,2)) / max_magnitude;

for i = 1:size(other_corr, 1)
    site_idx = other_corr(i, 1);
    corr_val = other_corr(i, 2);
    [x, y] = lattice.indexToCoord(site_idx);
    if corr_val >= 0
        color = positive_spin_color;
    else
        color = negative_spin_color;
    end
    scatter(x, y, marker_sizes(i), color, 'filled', 'MarkerEdgeColor', 'k');
end

plot(x_ref, y_ref, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'k');

if isempty(other_corr)
    real_max_abs = 0;
else
    real_max_abs = max(abs(other_corr(:,2)));
end

legend_max_abs = str2double(sprintf('%.1g', real_max_abs));
if legend_max_abs == 0
    legend_max_abs = 1;
end

legend_values = legend_max_abs * [-1, -0.5, -0.1, 0.1, 0.5, 1];

ax = gca;
x_lim = ax.XLim;
y_lim = ax.YLim;
plot_width = x_lim(2) - x_lim(1);
plot_height = y_lim(2) - y_lim(1);

legend_y = y_lim(1) + plot_height * 0.12;
text_y   = legend_y - plot_height * 0.06;
legend_x0 = x_lim(1) + 0.5 *plot_width;
dx = plot_width * 0.08;

for k = 1:numel(legend_values)
    vx = legend_x0 + (k-1) * dx;
    vv = legend_values(k);
    sz = base_marker_size * abs(vv) / real_max_abs;
    if sz <= 0
        sz = 1;
    end

    if vv < 0
        scatter(vx, legend_y, sz, negative_spin_color, 'filled', 'MarkerEdgeColor', 'k');
    else
        scatter(vx, legend_y, sz, positive_spin_color, 'filled', 'MarkerEdgeColor', 'k');
    end

    text(vx, text_y, sprintf('%.1g', vv), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', 'FontSize', 14, 'FontName', 'Arial');
end

param_str = sprintf('L_x = %d\nL_y = %d\nt'' = %.1ft\nJ_H = %.1ft\nU = %.1ft', Lx, Ly, t2, -Jk, U);
annotation('textbox', [0.2, 0.80, 0.18, 0.16], 'String', param_str, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'k', ...
    'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold', 'Margin', 5);

hold off;
axis off;
axis equal;

try
    set(gcf, 'Renderer','painters');
    if transparent_background
        set(gcf, 'Color','none', 'InvertHardcopy','off');
        set(findall(gcf, 'Type','axes'), 'Color','none');
    else
        set(gcf, 'Color',[1 1 1], 'InvertHardcopy','on');
    end
    this_file = mfilename('fullpath');
    if isempty(this_file)
        this_dir = pwd;
    else
        this_dir = fileparts(this_file);
    end
    fig_dir = fullfile(this_dir, 'figures');
    if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

    jh = -Jk;
    name_tokens = {
        kv_token('jh', jh, false), ...
        kv_token('t2', t2, true), ...
        kv_token('u',  U,  false), ...
        kv_token('ly', Ly, false), ...
        kv_token('lx', Lx, false)
    };
    base_name = ['kondo_ladder_spin_corr_', strjoin(name_tokens, '_')];
    pdf_path = fullfile(fig_dir, [base_name, '.pdf']);
    eps_path = fullfile(fig_dir, [base_name, '.eps']);

    if transparent_background
        exportgraphics(gcf, pdf_path, 'ContentType','vector', 'BackgroundColor','none');
    else
        exportgraphics(gcf, pdf_path, 'ContentType','vector');
    end
    print(gcf, '-depsc', '-painters', '-r600', eps_path);
catch ME
    warning(ME.identifier, '%s', ME.message);
end

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

function path = resolve_corr_path(base_dir, prefix, postfix, ly)
    candidate = fullfile(base_dir, [prefix, postfix]);
    if exist(candidate, 'file')
        path = candidate;
        return;
    end
    if ly ~= 2
        error('Correlation file not found for Ly=%d: %s', ly, candidate);
    end
    fallback = strrep(candidate, ['Ly', num2str(ly)], '');
    if exist(fallback, 'file')
        path = fallback;
        return;
    end
    error('Correlation file missing: tried %s and %s', candidate, fallback);
end

function postfix = build_postfix(t2, Jk, U, Ly, Lx, Db)
    parts = {
        ['t2', num2str(t2)], ...
        ['Jk', num2str(Jk)], ...
        ['U', num2str(U)], ...
        ['Ly', num2str(Ly)], ...
        ['Lx', num2str(Lx)], ...
        ['D', num2str(Db)], ...
        '.json'
    };
    postfix = strjoin(parts, '');
end