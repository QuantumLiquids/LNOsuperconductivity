function plot_charge_corr_extrapolation(extrapolation_order)
% plot_charge_corr_extrapolation.m
%
% Purpose
%   Extrapolate connected charge correlations C(d) = <n_i n_j> - <n_i><n_j>
%   to TE→0 across multiple bond dimensions D, for Δy=0 (intralayer) pairs.
%   Use quadratic (default) or linear extrapolation vs truncation error.
%
% Behavior
%   - Collect all available D by listing nfnf files with matching physical params
%   - For each D, compute C(d) with Δy=0 and integer d = |i-j|/8 (exclude d=0)
%   - Intersect common distances across D
%   - Extrapolate each distance to TE→0 using polyfit of specified order
%   - Plot loglog raw-by-D and extrapolated with power-law fit; print K_c
%
% Mapping (two-layer, two-orbital, two-leg): 8 sites per x; Δy=0 iff mod(|i-j|,8)==0

if nargin < 1
    extrapolation_order = 2; % default to quadratic
end

clc; close all;

% ---- Fixed Physical Parameters ----
Lx = 50;
Jk = -4;
Jperp = 4;
U = 18;

data_dir = '../../data/';

% ---- Enumerate available D via nfnf files ----
if Jperp == round(Jperp)
    base_pattern = sprintf('nfnfconventional_squareJk%dJperp%dU%dLx%dD*.json', Jk, Jperp, U, Lx);
else
    base_pattern = sprintf('nfnfconventional_squareJk%dJperp%.1fU%dLx%dD*.json', Jk, Jperp, U, Lx);
end
files = dir(fullfile(data_dir, base_pattern));
if isempty(files)
    error('No nfnf files found for the specified physical parameters.');
end

% Extract and sort D
d_values = zeros(numel(files), 1);
for i = 1:numel(files)
    s = files(i).name;
    d_str = regexp(s, 'D(\d+)\.json', 'tokens');
    if ~isempty(d_str)
        d_values(i) = str2double(d_str{1}{1});
    end
end
[sorted_d_values, sort_idx_d] = sort(d_values);
files = files(sort_idx_d);

% ---- Keep only D >= 10000 ----
min_D = 10000;
keep_D = sorted_d_values >= min_D;
files = files(keep_D);
sorted_d_values = sorted_d_values(keep_D);
if isempty(files)
    error('No nfnf files with D >= %d found for the specified physical parameters.', min_D);
end

% ---- Load truncation errors ----
all_truncation_errors = load_all_truncation_errors(data_dir, Jperp);
if isempty(all_truncation_errors)
    warning('No truncation error CSV found; falling back to 1/D.');
end

% ---- Compute C(d) per D ----
charge_correlations_by_d = cell(numel(files), 1);
truncation_errors = zeros(numel(files), 1);

for i = 1:numel(files)
    D = sorted_d_values(i);

    % Compose filenames for this D
    if Jperp == round(Jperp)
        file_postfix = sprintf('conventional_squareJk%dJperp%dU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
    else
        file_postfix = sprintf('conventional_squareJk%dJperp%.1fU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
    end

    fname_nfnf = fullfile(data_dir, ['nfnf', file_postfix]);
    fname_nf   = fullfile(data_dir, ['nf_local', file_postfix]);

    try
        [pairs_i, pairs_j, nfnf_val] = load_nfnf_pairs(fname_nfnf);
        [site_idx, site_val] = load_nf_local_series(fname_nf);

        % Build map for <n_i>
        nf_map = containers.Map('KeyType','double','ValueType','double');
        for k = 1:numel(site_idx)
            nf_map(site_idx(k)) = site_val(k);
        end

        % Δy=0 filter and distance
        delta_idx = abs(pairs_j - pairs_i);
        same_row = mod(delta_idx, 8) == 0;
        di = pairs_i(same_row);
        dj = pairs_j(same_row);
        vv = nfnf_val(same_row);
        dd = (delta_idx(same_row)) / 8;

        % Keep distances within (0, Lx/2]
        keep = (dd > 0) & (dd <= Lx/2);
        di = di(keep); dj = dj(keep); vv = vv(keep); dd = dd(keep);

        % Compute connected C_ij
        Cij = nan(size(dd));
        for t = 1:numel(dd)
            if isKey(nf_map, di(t)) && isKey(nf_map, dj(t))
                Cij(t) = vv(t) - nf_map(di(t)) * nf_map(dj(t));
            end
        end
        ok = ~isnan(Cij);
        dd = dd(ok); Cij = Cij(ok);

        % Aggregate by integer distance
        [u_d_all, ~, ic] = unique(dd);
        u_val_all = accumarray(ic, Cij, [], @mean);
        [u_d, ord] = sort(u_d_all);
        u_val = u_val_all(ord);

        charge_correlations_by_d{i} = struct('distances', u_d, 'correlations', u_val);

        % Truncation error for this D
        if ~isempty(all_truncation_errors)
            d_idx = find(all_truncation_errors.Bond_Dimension == D, 1);
            if ~isempty(d_idx)
                truncation_errors(i) = all_truncation_errors.Last_Sweep_Truncation_Error(d_idx);
            else
                truncation_errors(i) = 1 / D;
            end
        else
            truncation_errors(i) = 1 / D;
        end

    catch ME
        warning('Skipping D=%d due to error: %s', D, ME.message);
        charge_correlations_by_d{i} = [];
        truncation_errors(i) = NaN;
    end
end

% Filter out any empty entries
valid = ~cellfun(@isempty, charge_correlations_by_d) & ~isnan(truncation_errors);
charge_correlations_by_d = charge_correlations_by_d(valid);
sorted_d_values = sorted_d_values(valid);
truncation_errors = truncation_errors(valid);

if isempty(charge_correlations_by_d)
    error('No valid charge correlation datasets were loaded.');
end

% ---- Find common distances ----
common_distances = charge_correlations_by_d{1}.distances;
for i = 2:numel(charge_correlations_by_d)
    common_distances = intersect(common_distances, charge_correlations_by_d{i}.distances);
end

% ---- Extrapolate per distance ----
extrapolated_correlations = NaN(size(common_distances));

for d_idx = 1:length(common_distances)
    distance = common_distances(d_idx);
    corr_at_distance = [];
    te_at_distance = [];
    for i = 1:numel(charge_correlations_by_d)
        dist_idx = find(charge_correlations_by_d{i}.distances == distance, 1);
        if ~isempty(dist_idx)
            corr_at_distance(end+1) = charge_correlations_by_d{i}.correlations(dist_idx); %#ok<AGROW>
            te_at_distance(end+1) = truncation_errors(i); %#ok<AGROW>
        end
    end

    if numel(corr_at_distance) >= 2
        ord = min(max(round(extrapolation_order), 1), 2); % clamp to 1 or 2
        p = polyfit(te_at_distance, corr_at_distance, ord);
        extrapolated_correlations(d_idx) = polyval(p, 0);
    end
end

% ---- Plotting ----
figure('Position', [100, 100, 1200, 500]);

% Subplot 1: raw |C(d)| for each D
subplot(1,2,1);
hold on; box on; grid on;
set(gca, 'XScale', 'log', 'YScale', 'log', 'FontName', 'Arial');
colors = lines(numel(charge_correlations_by_d));
for i = 1:numel(charge_correlations_by_d)
    D = sorted_d_values(i);
    color = colors(i, :);
    loglog(charge_correlations_by_d{i}.distances, abs(charge_correlations_by_d{i}.correlations), ...
           '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', color, 'DisplayName', ['D = ', num2str(D)]);
end
xlabel('Distance |x_i - x_j|', 'FontSize', 14);
ylabel('|Charge Corr|', 'FontSize', 14);
title('Raw charge correlations vs D', 'FontSize', 16);
legend('show', 'Location', 'northeast');
set(gca, 'FontSize', 12);

% Subplot 2: TE→0 extrapolated and power-law fit
subplot(1,2,2);
hold on; box on; grid on;
set(gca, 'XScale', 'log', 'YScale', 'log', 'FontName', 'Arial');
loglog(common_distances, abs(extrapolated_correlations), 'o-', 'Color', [0.2 0.2 0.7], ...
       'LineWidth', 3, 'MarkerSize', 10, 'DisplayName', 'Extrapolated (TE \rightarrow 0)');

% Power law fit |C(r)| = A r^{-K_c} on profile distances
profile_r = 3:2:25;
valid_idx = ~isnan(extrapolated_correlations) & (extrapolated_correlations ~= 0) & ismember(common_distances, profile_r);
if sum(valid_idx) >= 2
    log_r = log(common_distances(valid_idx));
    log_C = log(abs(extrapolated_correlations(valid_idx)));
    p = polyfit(log_r, log_C, 1);
    K_c = -p(1);
    A = exp(p(2));

    % Extend fit curve
    ax2 = gca; r_min = max(min(common_distances(valid_idx)), ax2.XLim(1));
    r_max = max([ax2.XLim(2), max(common_distances)]);
    fit_r = logspace(log10(r_min), log10(r_max), 200);
    fit_C = A * fit_r.^(-K_c);
    loglog(fit_r, fit_C, 'b--', 'LineWidth', 2, 'DisplayName', sprintf('Fit: %.3f/r^{K_c=%.3f}', A, K_c));
    title_str = sprintf('Extrapolated charge corr (order %d)\nPower law: %.3f/r^{K_c=%.3f}', extrapolation_order, A, K_c);
else
    title_str = sprintf('Extrapolated charge corr (order %d)', extrapolation_order);
    K_c = NaN;
end

xlabel('Distance |x_i - x_j|', 'FontSize', 14);
ylabel('|Charge Corr|', 'FontSize', 14);
title(title_str, 'FontSize', 16);
legend('show', 'Location', 'southwest');
set(gca, 'FontSize', 12);

sgtitle(sprintf('Charge correlation extrapolation (J_K=%d, J_{perp}=%g, U=%d, Lx=%d, order %d)', ...
        Jk, Jperp, U, Lx, extrapolation_order), 'FontSize', 18, 'FontName', 'Arial');

% Save figure
if Jperp == round(Jperp)
    ord_str = sprintf('%d', extrapolation_order);
    filename_base = sprintf('charge_corr_extrapolation_Jk%dJperp%dU%dLx%d_%sorder', Jk, Jperp, U, Lx, ord_str);
else
    ord_str = sprintf('%d', extrapolation_order);
    filename_base = sprintf('charge_corr_extrapolation_Jk%dJperp%.1fU%dLx%d_%sorder', Jk, Jperp, U, Lx, ord_str);
end
saveas(gcf, [filename_base, '.png']);
saveas(gcf, [filename_base, '.pdf']);

% Summary prints
fprintf('\n=== Charge Correlation Extrapolation ===\n');
fprintf('Physical parameters: J_K=%d, J_perp=%g, U=%d, Lx=%d\n', Jk, Jperp, U, Lx);
fprintf('D values used: [%s]\n', num2str(sorted_d_values));
fprintf('Truncation errors: [%s]\n', num2str(truncation_errors, 4));
fprintf('Common distances: %s\n', mat2str(common_distances));
if exist('K_c','var') && ~isnan(K_c)
    fprintf('Fitted K_c: %.6f\n', K_c);
end

end

% ============================================================================
% Helpers
% ============================================================================
function all_te = load_all_truncation_errors(data_dir, Jperp)
    if Jperp == round(Jperp)
        te_filename = sprintf('201_202_bond_last_sweep_errors_Jperp%d.csv', Jperp);
    else
        te_filename = sprintf('201_202_bond_last_sweep_errors_Jperp%.1f.csv', Jperp);
    end
    te_file_path = fullfile(data_dir, te_filename);
    if exist(te_file_path, 'file')
        try
            all_te = readtable(te_file_path);
        catch
            all_te = [];
        end
    else
        all_te = [];
    end
end

function [i_list, j_list, val_list] = load_nfnf_pairs(fname)
    if ~exist(fname, 'file')
        error('File not found: %s', fname);
    end
    raw = jsondecode(fileread(fname));
    if iscell(raw)
        N = numel(raw);
        i_list = zeros(N,1); j_list = zeros(N,1); val_list = zeros(N,1);
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
                v = v(1);
            end
            val_list(k) = v;
        end
    else
        if size(raw,2) < 3
            error('Unexpected nfnf JSON shape.');
        end
        i_list = raw(:,1); j_list = raw(:,2); val_list = raw(:,3);
    end
end

function [indices, values] = load_nf_local_series(fname)
    if ~exist(fname, 'file')
        error('File not found: %s', fname);
    end
    raw = jsondecode(fileread(fname));
    if iscell(raw)
        N = numel(raw);
        indices = zeros(N,1); values = zeros(N,1);
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
        if size(raw,2) < 2
            error('Unexpected nf_local JSON shape.');
        end
        indices = raw(:,1); values = raw(:,2);
    end
end


