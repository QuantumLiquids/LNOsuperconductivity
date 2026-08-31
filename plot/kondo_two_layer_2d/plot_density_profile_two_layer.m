% plot_density_profile_two_layer.m
%
% Purpose
%   Load local density `nf_local...json` containing only itinerant sites
%   (even site indices), group every 4 values per x (2 layers × 2 legs),
%   check intra-x consistency within 1%, average them, and plot density vs x.
%
% Data format
%   JSON array of pairs: [ [site_idx], value ] or [ [site_idx], [Re, Im] ].
%   Site indices start at 0, with 8 physical sites per x-position.
%   Even indices enumerate itinerant sites. For each x, the 4 itinerant
%   indices are 8*x + {0, 2, 4, 6} for x starting from 0.
%
% Checks
%   - Verify all indices are even; warn if any odd index is present
%   - For each x, the 4 values must agree within 1% relative deviation.
%     If violated, print a warning with x and values.
%
clear; close all;

% ---- Physical / file parameters (match spin-corr script) ----
Lx     = 50;
Ly     = 2; % comment only
Jk     = -4;
Jperp  = 4;
U      = 18;
D      = 20000;

data_dir = '../../data/';

% ---- Build filename postfix ----
if Jperp == round(Jperp)
    file_postfix = sprintf('conventional_squareJk%dJperp%dU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
else
    file_postfix = sprintf('conventional_squareJk%dJperp%.1fU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
end

fname = fullfile(data_dir, ['nf_local', file_postfix]);

% ---- Load and parse ----
[site_idx, site_val] = load_nf_local_series(fname);

% Sort by site index for predictable grouping
[site_idx, ord] = sort(site_idx(:));
site_val = site_val(ord);

% ---- Basic sanity checks ----
if any(mod(site_idx, 2) ~= 0)
    warning('Found odd site indices in nf_local (expected only even indices for itinerant sites).');
end

% Map each site to x-position via stride-8 layout
x_pos = floor(site_idx / 8) + 1; % 1..Lx
Lx_detected = max(x_pos);
if Lx_detected ~= Lx
    warning('Detected Lx=%d from indices, which differs from specified Lx=%d.', Lx_detected, Lx);
end

% ---- Group every 4 itinerant sites per x and check 1% consistency ----
x_list = 1:Lx_detected;
avg_density = zeros(size(x_list));
tol_rel = 0.01; % 1%

for x = x_list
    mask = (x_pos == x);
    vals = site_val(mask);

    if numel(vals) ~= 4
        warning('x=%d has %d itinerant entries (expected 4).', x, numel(vals));
    end

    if isempty(vals)
        avg_density(x) = NaN;
        continue;
    end

    mu = mean(vals);
    denom = max(abs(mu), 1e-20);
    rel_dev = max(abs(vals - mu)) / denom;
    if rel_dev > tol_rel
        vals4 = NaN(1,4);
        c = min(4, numel(vals));
        vals4(1:c) = vals(1:c);
        warning('Density inconsistency >1%% at x=%d: values = [%g, %g, %g, %g], rel dev = %.3f%%', ...
                x, vals4(1), vals4(2), vals4(3), vals4(4), 100*rel_dev);
    end

    avg_density(x) = mu;
end

% ---- Plot density profile ----
figure('Position', [120, 120, 800, 500]);
hold on; box on; grid on;

plot(x_list, avg_density, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
     'DisplayName', 'itinerant density (avg over layers, Ly=2)');

xlim([1, Lx_detected]);
xlabel('x position', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Density n(x)', 'FontSize', 18, 'FontWeight', 'bold');
title(sprintf('Density profile (J_K=%g, J_\\perp=%g, U=%g, L_x=%d, D=%d)', ...
      Jk, Jperp, U, Lx, D), 'FontSize', 18, 'FontWeight', 'bold');
legend('show', 'Location', 'best', 'FontSize', 12);

ax = gca; ax.FontSize = 14; ax.FontWeight = 'bold';


% ---- Helpers ----
function [indices, values] = load_nf_local_series(fname)
    if ~exist(fname, 'file')
        error('File not found: %s', fname);
    end

    raw = jsondecode(fileread(fname));

    if iscell(raw)
        % Cell array: raw{k} = { [idx], val } or { [idx], [Re, Im] }
        N = numel(raw);
        indices = zeros(N, 1);
        values  = zeros(N, 1);
        for k = 1:N
            item = raw{k};
            % First element: index vector (use first component)
            idx_part = item{1};
            if iscell(idx_part)
                idx_vec = cell2mat(idx_part);
            else
                idx_vec = idx_part;
            end
            indices(k) = idx_vec(1);

            % Second element: scalar value or [Re, Im]
            val_part = item{2};
            if iscell(val_part)
                v = cell2mat(val_part);
            else
                v = val_part;
            end
            if ~isscalar(v)
                v = v(1);
            end
            values(k) = v;
        end
    else
        % Numeric matrix: columns are [index, value]
        if size(raw, 2) < 2
            error('Unexpected JSON shape. Expect Nx2 numeric or cell array of pairs.');
        end
        indices = raw(:, 1);
        values  = raw(:, 2);
    end
end



