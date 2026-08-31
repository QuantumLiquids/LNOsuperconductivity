% plot_single_particle_corr_two_layer.m
%
% Purpose
%   Load four single-particle correlation datasets and plot intralayer
%   correlators versus integer Δx for the two-layer Kondo model.
%   Files:
%     - cup_dag_cup...json
%     - cup_cup_dag...json
%     - cdown_dag_cdown...json
%     - cdown_cdown_dag...json
%
% Mapping and filtering
%   - Even indices are itinerant electrons; odd are localized (see vmps_conventional_square.cpp).
%   - There are 8 physical sites per x-position: 2 legs (Ly=2) × 2 layers × 2 dof.
%   - We keep strictly Δy = 0 intralayer bonds: mod(i - i_ref, 8) == 0.
%   - Integer distance along x is |i - i_ref| / 8.
%   - Data use a single reference row (single i_ref); no averaging over the two y rows.
%
% Output
%   - One figure with the sum of the four intralayer series vs integer Δx.

clear; close all;

% ---- Physical / file parameters (edit as needed) ----
Lx     = 50;
Ly     = 2; %used for comments; mapping already encodes this
Jk     = -4;
Jperp  = 4;  % integer or fractional; formatting handled below
U      = 18;
D      = 20000;

data_dir = '../../data/';

% ---- Build filename postfix ----
if Jperp == round(Jperp)
    file_postfix = sprintf('conventional_squareJk%dJperp%dU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
else
    file_postfix = sprintf('conventional_squareJk%dJperp%.1fU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
end

files = {
    ['cup_dag_cup', file_postfix];
    ['cup_cup_dag', file_postfix];
    ['cdown_dag_cdown', file_postfix];
    ['cdown_cdown_dag', file_postfix]
};


% ---- Load and filter intralayer data ----
all_dist = cell(1, numel(files));
all_val  = cell(1, numel(files));

for k = 1:numel(files)
    fname = fullfile(data_dir, files{k});
    if ~exist(fname, 'file')
        error('File not found: %s', fname);
    end
    raw = jsondecode(fileread(fname));
    % raw is a cell array: { { [i_ref, j_ref], [Re, Im] }, { [i_ref, j], [Re, Im] }, ... }
    % Use first entry to get reference index.
    ref_pair = raw{1}{1};
    ref_i = ref_pair(1);

    target_i = zeros(numel(raw), 1);
    vals     = zeros(numel(raw), 1);
    for n = 1:numel(raw)
        pair = raw{n}{1};
        % pair is [i_ref, j]
        target_i(n) = pair(2);
        val_ri = raw{n}{2};
        vals(n) = val_ri(1); % real part only (imag part should be ~0)
    end

    delta_idx = abs(target_i - ref_i);
    same_row = mod(delta_idx, 8) == 0; % intralayer, same y
    dist = delta_idx(same_row) / 8;    % integer Δx
    v    = vals(same_row);

    % Exclude Δx == 0 entirely (covers (Δx=0,Δy=0); the (Δx=0,Δy=1) is not in same_row)
    nz = dist > 0;
    dist = dist(nz);
    v    = v(nz);

    % Sort by distance
    [dist_sorted, idx] = sort(dist);
    v_sorted = v(idx);

    all_dist{k} = dist_sorted(:);
    all_val{k}  = v_sorted(:);
end

% ---- Aggregate per Δx (average over ±Δx duplicates on the same y row) ----
uniq_dist = cell(1, numel(files));
uniq_val  = cell(1, numel(files));
for k = 1:numel(files)
    [u_d, ~, ic] = unique(all_dist{k});
    u_v = accumarray(ic, all_val{k}, [], @mean);
    keep = (u_d <= Lx/2);
    uniq_dist{k} = u_d(keep);
    uniq_val{k}  = u_v(keep);
end

% ---- Intersect Δx across all four series ----
common_d = uniq_dist{1};
for k = 2:numel(files)
    common_d = intersect(common_d, uniq_dist{k});
end
common_d = common_d(common_d > 0 & common_d <= Lx/2);
common_d = sort(common_d);

if isempty(common_d)
    error('No common Δx values across four series up to Lx/2.');
end

% ---- Align values to common Δx ----
V = zeros(numel(common_d), numel(files));
for k = 1:numel(files)
    [~, loc] = ismember(common_d, uniq_dist{k});
    V(:, k) = uniq_val{k}(loc);
end

% ---- Similarity check: same sign and within 5% spread on ≥95% points ----
sign_ref = sign(V(:, 1));
same_sign = all(bsxfun(@eq, sign(V), sign_ref), 2);
den = max(abs(V), [], 2);
rel_spread = zeros(size(den));
nonzero = den > 0;
rel_spread(nonzero) = (max(abs(V(nonzero, :)), [], 2) - min(abs(V(nonzero, :)), [], 2)) ./ den(nonzero);
ok = same_sign & (rel_spread <= 0.05);
ok_ratio = mean(ok);
if ok_ratio < 0.95
    warning('Similarity check failed: %.1f%% points within 5%% band and same sign (threshold 95%%).', 100*ok_ratio);
else
    fprintf('Similarity check passed: %.1f%% points within 5%% band and same sign.\n', 100*ok_ratio);
end

% ---- Sum the four series, then divide by 2 to average Hermitian conjugates ----
% This yields spin-summed <c^\dag c> (up + down), using <c c^\dag> = <c^\dag c>.
sum_series = sum(V, 2) / 2;

figure('Position', [120, 120, 800, 600]);
hold on; box on; grid on;
set(gca,  'YScale', 'log');
plot(common_d, abs(sum_series), '-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Spin-summed <c^\dag c> (Δy=0 intralayer)');
xlim([1, Lx/2]);

xlabel('|\\Delta x|', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Single-particle |G(i,j)| (sum of 4, Δy=0)', 'FontSize', 18, 'FontWeight', 'bold');
title(sprintf('Intralayer (Δy=0) single-particle correlations, sum of 4 (J_K=%g, J_\\perp=%g, U=%g, L_x=%d, D=%d)', ...
      Jk, Jperp, U, Lx, D), 'FontSize', 18, 'FontWeight', 'bold');
legend('show', 'Location', 'northeast', 'FontSize', 12);

ax = gca; ax.FontSize = 14; ax.FontWeight = 'bold';



% ---- Exponential fit on selected distances (7:2:25) ----
dsel_list = 7:2:25;
mask_fit = ismember(common_d, dsel_list);
d_fit_data = common_d(mask_fit);
y_fit_data = abs(sum_series(mask_fit));

% Ensure strictly positive values for log-fit
pos = y_fit_data > 0;
d_fit_data = d_fit_data(pos);
y_fit_data = y_fit_data(pos);

if numel(d_fit_data) < 2
    warning('Not enough positive points for fitting within distances 7:2:25.');
else
    p = polyfit(d_fit_data, log(y_fit_data), 1); % log(y) = log(A) - x/xi
    xi = -1 / p(1);
    A  = exp(p(2));

    d_fit_line = d_fit_data(1):d_fit_data(end);
    y_fit_line = A * exp(-d_fit_line / xi);
    plot(d_fit_line, y_fit_line, 'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('Fit: \\xi=%.3f', xi));

    fprintf('Exponential fit on distances Δx = 7:2:25 -> correlation length xi = %.6f\n', xi);
end
