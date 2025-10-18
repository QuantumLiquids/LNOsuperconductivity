% plot_kondo_two_leg/plot_spin_corr_two_panel.m
%
% Purpose
%   Visualize equal-time spin correlations for TWO parameter sets side-by-side
%   on the two-leg tilted Kondo lattice (single-layer model). A unified bubble
%   legend is drawn once and shared by both subplots. Marker size encodes
%   |corr| and color encodes sign.
%
% Inputs (configured in-file)
%   params(1), params(2) each specify L, t2, Jk, U, Db used to form the data
%   file postfix and annotate panels. Db defaults to 0 if you are unsure.
%
% Data dependencies
%   Reads JSON from ../../data/ with names:
%     szsz<t2...Jk...U...Lx...D...>.json
%     spsm<t2...Jk...U...Lx...D...>.json
%     smsp<t2...Jk...U...Lx...D...>.json
%
% Other dependencies
%   Requires TiltedZigZagLattice.m in the same folder for geometry.
clear;
close all;

% -----------------------------------------------------------------------------
% Parameter sets (edit these as needed)
% NOTE: J_H = -Jk in the figure annotation.
params(1).Lx  = 20; params(1).Ly = 2; params(1).t2 = 0.3; params(1).Jk = -4; params(1).U = 14.0; params(1).Db = 0;
params(2).Lx  = 20; params(2).Ly = 4; params(2).t2 = 0.3; params(2).Jk = -4; params(2).U = 14.0; params(2).Db = 12000;

base_marker_size = 300;  % Max size for largest magnitude

% Colors (light purple for positive, light red for negative)
positive_spin_color = [142 139 254]/256;
negative_spin_color = [232 132 130]/256;

% -----------------------------------------------------------------------------
% Create lattice (assumes same L for both; if different, the code adapts per panel)
% We'll instantiate per panel to match each L.

% Preload both panels' correlation data to compute a GLOBAL size scale
panel_data = cell(1,2);
global_max_abs = 0;
ref_site_idx_all = zeros(1,2);
lat_per_panel = cell(1,2);

for p = 1:2
    Lxp = params(p).Lx;
    Lyp = params(p).Ly;
    lat_per_panel{p} = TiltedZigZagLattice(Lyp, Lxp, 'OBC');
    [corr_data, ref_site_idx, other_corr] = load_corr_data(params(p));
    panel_data{p}.corr_data = corr_data;
    panel_data{p}.other_corr = other_corr;
    ref_site_idx_all(p) = ref_site_idx;
    if ~isempty(other_corr)
        global_max_abs = max(global_max_abs, max(abs(other_corr(:,2))));
    end
end

if global_max_abs == 0
    global_max_abs = 1; % avoid divide-by-zero in degenerate case
end

% -----------------------------------------------------------------------------
% Layout: two panels with tight spacing
tl = tiledlayout(1,2, 'TileSpacing','compact', 'Padding','compact');

for p = 1:2
    nexttile(p);
    lattice = lat_per_panel{p};
    corr_data = panel_data{p}.corr_data;
    other_corr = panel_data{p}.other_corr;
    ref_site_idx = ref_site_idx_all(p);

    lattice.drawLattice(1.5, 0);
    hold on;

    % Reference site star
    [x_ref, y_ref] = lattice.indexToCoord(ref_site_idx);
    plot(x_ref, y_ref, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'k');

    % Marker sizes scaled by GLOBAL max across both panels
    if isempty(other_corr)
        marker_sizes = [];
    else
        marker_sizes = base_marker_size * abs(other_corr(:,2)) / global_max_abs;
    end

    % Plot other sites
    for i = 1:size(other_corr,1)
        site_idx = other_corr(i,1);
        corr_val = other_corr(i,2);
        [x, y] = lattice.indexToCoord(site_idx);
        if corr_val >= 0
            color = positive_spin_color;
        else
            color = negative_spin_color;
        end
        scatter(x, y, marker_sizes(i), color, 'filled', 'MarkerEdgeColor', 'k');
    end

    % Axis styling
    axis equal; axis off;

    % Panel label: (a), (b)
    ax = gca;
    x_lim = ax.XLim; y_lim = ax.YLim;
    plot_width = x_lim(2) - x_lim(1);
    plot_height = y_lim(2) - y_lim(1);
    panel_labels = {'(a)', '(b)'};
    label_text = panel_labels{p};
    if ~isempty(label_text)
        text(x_lim(1) + 0.02*plot_width, y_lim(2) - 0.06*plot_height, ...
            label_text, 'FontWeight','bold', 'FontSize', 16, ...
            'FontName','Arial');
    end

    % Parameters box in each panel
    t2 = params(p).t2; Jk = params(p).Jk; U = params(p).U;
    param_str = sprintf('t'' = %.1ft\nJ_H = %.1ft\nU = %.1ft', t2, -Jk, U);
    % Place near top-left inside axes
    text(x_lim(1) + 0.18*plot_width, y_lim(2) - 0.10*plot_height, param_str, ...
        'HorizontalAlignment','left', 'VerticalAlignment','top', ...
        'FontName','Arial', 'FontSize', 14, 'FontWeight','bold', ...
        'BackgroundColor','white', 'Margin', 4, 'EdgeColor','k');

    hold off;
end

% -----------------------------------------------------------------------------
% Unified bubble legend drawn once in a small, invisible axes below panels
% Compute one-significant-digit legend values using GLOBAL max
legend_max_abs = str2double(sprintf('%.1g', global_max_abs));
if legend_max_abs == 0
    legend_max_abs = 1;
end
legend_values = legend_max_abs * [-1, -0.5, -0.1, 0.1, 0.5, 1];

% Create a dedicated legend axes spanning width under tiles
% Position relative to the tiledlayout
outerpos = tl.OuterPosition; % [x y w h] in normalized figure units
fig = gcf;

% Legend axes rectangle
leg_ax_height = 0.12 * outerpos(4);
leg_ax_y = max(0.03, outerpos(2) - 0.05);
leg_ax = axes('Position', [outerpos(1) + 0.10*outerpos(3), leg_ax_y, 0.80*outerpos(3), leg_ax_height]);
hold(leg_ax, 'on');
axis(leg_ax, 'off');
% Fix limits and avoid equal aspect so labels are visible
xlim(leg_ax, [0 1]); ylim(leg_ax, [0 1]);

% Layout bubbles horizontally
ax = leg_ax;
x_lim = [0 1]; y_lim = [0 1];
legend_y = 0.65;            % baseline for circles
text_y   = 0.28;            % text slightly below
dx = 0.12;                  % spacing
% center horizontally
n_bubbles = numel(legend_values);
legend_x0 = 0.5 - 0.5*(n_bubbles-1)*dx;
legend_x0 = max(0.05, min(legend_x0, 0.95 - (n_bubbles-1)*dx));

for k = 1:numel(legend_values)
    vx = legend_x0 + (k-1) * dx;
    vv = legend_values(k);
    sz = base_marker_size * abs(vv) / global_max_abs;
    if sz <= 0
        sz = 1;
    end
    if vv < 0
        scatter(ax, vx, legend_y, sz, negative_spin_color, 'filled', 'MarkerEdgeColor','k');
    else
        scatter(ax, vx, legend_y, sz, positive_spin_color, 'filled', 'MarkerEdgeColor','k');
    end
    text(ax, vx, text_y, sprintf('%.1g', vv), 'HorizontalAlignment','center', ...
        'VerticalAlignment','top', 'FontSize', 12, 'FontName','Arial');
end

% Optional legend title
text(ax, legend_x0 - 0.06, legend_y, '', 'HorizontalAlignment','right', ...
    'FontSize', 12, 'FontName','Arial');

hold(ax, 'off');


% =============================================================================
% Helpers
function [corr_data, ref_site_idx, other_corr] = load_corr_data(p)
    postfix = build_postfix(p.t2, p.Jk, p.U, p.Ly, p.Lx, p.Db);
    base_path = '../../data/';
    SpinCorrDataZZ = jsondecode(fileread(resolve_corr_path(base_path, 'szsz', postfix, p.Ly)));
    SpinCorrDataPM = jsondecode(fileread(resolve_corr_path(base_path, 'spsm', postfix, p.Ly)));
    SpinCorrDataMP = jsondecode(fileread(resolve_corr_path(base_path, 'smsp', postfix, p.Ly))); %#ok<NASGU>

    data_num = numel(SpinCorrDataZZ);
    ref_site_raw = SpinCorrDataZZ{1}{1}(1);
    target_site_idx = zeros(1, data_num);
    SpinCorr = zeros(1, data_num);
    for i = 1:data_num
        target_site_idx(i) = SpinCorrDataZZ{i}{1}(2);
        SpinCorr(i) = SpinCorrDataZZ{i}{2} + SpinCorrDataPM{i}{2};
    end

    raw_indices = [ref_site_raw; target_site_idx'];
    if any(mod(raw_indices, 2) ~= 0)
        error('Spin correlation dataset includes localized-site indices (odd). Expected even indices for itinerant electrons.');
    end

    ref_site_idx = ref_site_raw / 2;
    site_indices = target_site_idx' / 2;

    corr_data = [site_indices, SpinCorr'];
    other_corr = corr_data(corr_data(:,1) ~= ref_site_idx, :);
end

% -----------------------------------------------------------------------------
% Transparent vector export to figures/
try
    set(gcf, 'Renderer','painters');
     set(gcf, 'Color','none', 'InvertHardcopy','off');
    set(findall(gcf, 'Type','axes'), 'Color','none');
    this_file = mfilename('fullpath');
    if isempty(this_file)
        this_dir = pwd;
    else
        this_dir = fileparts(this_file);
    end
    fig_dir = fullfile(this_dir, 'figures');
    if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

    g1 = strjoin({ kv_token('jh', -params(1).Jk, false), ...
                   kv_token('t2', params(1).t2, true), ...
                   kv_token('u',  params(1).U,  false), ...
                   kv_token('ly', params(1).Ly, false), ...
                   kv_token('lx', params(1).Lx, false) }, '_');
    g2 = strjoin({ kv_token('jh', -params(2).Jk, false), ...
                   kv_token('t2', params(2).t2, true), ...
                   kv_token('u',  params(2).U,  false), ...
                   kv_token('ly', params(2).Ly, false), ...
                   kv_token('lx', params(2).Lx, false) }, '_');
    base_name = ['kondo_ladder_spin_corr_two_params_', g1, '_and_', g2];
    pdf_path = fullfile(fig_dir, [base_name, '.pdf']);
    eps_path = fullfile(fig_dir, [base_name, '.eps']);

    % exportgraphics(gcf, pdf_path, 'ContentType','vector', 'BackgroundColor','none');
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


