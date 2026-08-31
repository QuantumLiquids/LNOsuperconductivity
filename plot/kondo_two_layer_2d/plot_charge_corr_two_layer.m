% plot_charge_corr_two_layer.m
%
% Purpose
%   Plot connected charge correlations C(d) = <n_i n_j> - <n_i><n_j>
%   for Δy = 0 (intralayer) bonds of the two-layer Kondo model, versus
%   integer Δx = |i - j| / 8. The <n_i n_j> data come from nfnf...json,
%   and the one-point <n_i> are loaded from nf_local...json without
%   any per-x averaging (use site-resolved density directly).
%
% Mapping
%   - Even indices are itinerant electrons; odd are localized.
%   - 8 physical sites per x-position: 2 legs × 2 layers × 2 dof.
%   - Intralayer, Δy = 0 bonds: mod(|i - j|, 8) == 0.
%   - Distance along x is |i - j| / 8, an integer.
%
clear; close all;

% ---- Physical / file parameters (edit as needed) ----
Lx     = 50;
Ly     = 2; % used in comments only
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

fname_nfnf    = fullfile(data_dir, ['nfnf', file_postfix]);
fname_nf_local = fullfile(data_dir, ['nf_local', file_postfix]);

% ---- Load two-point and one-point data ----
[pairs_i, pairs_j, nfnf_val] = load_nfnf_pairs(fname_nfnf);
[site_idx, site_val]         = load_nf_local_series(fname_nf_local);

% Optional: ensure we have only even indices for itinerant sites
if any(mod(site_idx, 2) ~= 0)
    warning('nf_local contains odd site indices (expected only even for itinerant sites).');
end

% Map site -> density for quick lookup
nf_map = containers.Map('KeyType','double','ValueType','double');
for k = 1:numel(site_idx)
    nf_map(site_idx(k)) = site_val(k);
end

% ---- Keep Δy=0 intralayer pairs; compute integer Δx ----
delta_idx = abs(pairs_j - pairs_i);
same_row = mod(delta_idx, 8) == 0;

pairs_i = pairs_i(same_row);
pairs_j = pairs_j(same_row);
nfnf_val = nfnf_val(same_row);
delta_idx = delta_idx(same_row);

dist = delta_idx / 8; % integer Δx

% Exclude Δx == 0
keep = (dist > 0) & (dist <= Lx/2);
pairs_i = pairs_i(keep);
pairs_j = pairs_j(keep);
nfnf_val = nfnf_val(keep);
dist     = dist(keep);

% ---- Compute connected correlator per pair: C_ij = <n_i n_j> - <n_i><n_j> ----
Cij = zeros(size(dist));
missing_count = 0;
for t = 1:numel(dist)
    i = pairs_i(t); j = pairs_j(t);
    if ~isKey(nf_map, i) || ~isKey(nf_map, j)
        % Missing density for i or j; skip
        Cij(t) = NaN;
        missing_count = missing_count + 1;
        continue;
    end
    Cij(t) = nfnf_val(t) - nf_map(i) * nf_map(j);
end
if missing_count > 0
    warning('Missing %d one-point entries in nf_local for some pairs; treated as NaN.', missing_count);
end

% Remove NaNs before aggregation
ok = ~isnan(Cij);
dist = dist(ok);
Cij  = Cij(ok);

% ---- Aggregate by integer distance (average over all pairs with same Δx) ----
[u_d_all, ~, ic] = unique(dist);
u_val_all = accumarray(ic, Cij, [], @mean);

% Keep up to Lx/2 (already applied), and sort
[u_d, ord] = sort(u_d_all);
u_val = u_val_all(ord);

% ---- Plot |C(d)| vs Δx on log scale ----
figure('Position', [120, 120, 800, 600]);
hold on; box on; grid on;
set(gca, 'YScale', 'log');

plot(u_d, abs(u_val), '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
     'DisplayName', '\langle n_i n_j \rangle - \langle n_i \rangle \langle n_j \rangle (\Delta y=0)');

xlim([1, Lx/2]);
xlabel('|\\Delta x|', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Charge corr |\langle n n \rangle - \langle n \rangle^2|', 'FontSize', 18, 'FontWeight', 'bold');
title(sprintf('Intralayer (\Delta y=0) charge correlations (J_K=%g, J_\\perp=%g, U=%g, L_x=%d, D=%d)', ...
      Jk, Jperp, U, Lx, D), 'FontSize', 18, 'FontWeight', 'bold');
legend('show', 'Location', 'northeast', 'FontSize', 12);

ax = gca; ax.FontSize = 14; ax.FontWeight = 'bold';


% ---- Helpers ----
function [i_list, j_list, val_list] = load_nfnf_pairs(fname)
    if ~exist(fname, 'file')
        error('File not found: %s', fname);
    end
    raw = jsondecode(fileread(fname));

    if iscell(raw)
        N = numel(raw);
        i_list = zeros(N,1);
        j_list = zeros(N,1);
        val_list = zeros(N,1);
        for k = 1:N
            item = raw{k};
            ij_part = item{1};
            if iscell(ij_part)
                ij = cell2mat(ij_part);
            else
                ij = ij_part;
            end
            i_list(k) = ij(1);
            j_list(k) = ij(2);

            val_part = item{2};
            if iscell(val_part)
                v = cell2mat(val_part);
            else
                v = val_part;
            end
            if ~isscalar(v)
                v = v(1); % take real part
            end
            val_list(k) = v;
        end
    else
        % Numeric matrix assumed: [i, j, value]
        if size(raw,2) < 3
            error('Unexpected nfnf JSON shape. Expect cell array of pairs or Nx3 numeric.');
        end
        i_list = raw(:,1);
        j_list = raw(:,2);
        val_list = raw(:,3);
    end
end

function [indices, values] = load_nf_local_series(fname)
    if ~exist(fname, 'file')
        error('File not found: %s', fname);
    end
    raw = jsondecode(fileread(fname));

    if iscell(raw)
        N = numel(raw);
        indices = zeros(N, 1);
        values  = zeros(N, 1);
        for k = 1:N
            item = raw{k};
            idx_part = item{1};
            if iscell(idx_part)
                idx_vec = cell2mat(idx_part);
            else
                idx_vec = idx_part;
            end
            indices(k) = idx_vec(1);

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
        if size(raw, 2) < 2
            error('Unexpected nf_local JSON shape. Expect Nx2 numeric or cell array of pairs.');
        end
        indices = raw(:, 1);
        values  = raw(:, 2);
    end
end


