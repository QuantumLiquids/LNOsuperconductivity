% plot_spin_corr_two_layer.m
%
% Purpose
%   Load three spin-correlation datasets and plot intralayer correlators
%   versus integer Δx for the two-layer Kondo model.
%   Files:
%     - szsz...json  (⟨S^z_i S^z_j⟩)
%     - smsp...json  (⟨S^-_i S^+_j⟩)
%     - spsm...json  (⟨S^+_i S^-_j⟩)
%
% Mapping and filtering
%   - Even indices are itinerant electrons; odd are localized (see vmps_conventional_square.cpp).
%   - There are 8 physical sites per x-position: 2 legs (Ly=2) × 2 layers × 2 dof.
%   - Keep strictly Δy = 0 intralayer bonds: mod(i - i_ref, 8) == 0.
%   - Integer distance along x is |i - i_ref| / 8.
%   - Data use a single reference row (single i_ref); no averaging over the two y rows.
%
% Consistency checks (performed on common Δx):
%   1) smsp ≈ spsm up to absolute error 1e-14
%   2) 2*szsz ≈ smsp up to 1% relative error

clear; close all;

% ---- Physical / file parameters (edit as needed) ----
Lx     = 50;
Ly     = 2; % used for comments; mapping already encodes this
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

files = struct();
files.szsz = ['szsz', file_postfix];
files.smsp = ['smsp', file_postfix];
files.spsm = ['spsm', file_postfix];


% ---- Load and filter three series ----
[d_szsz_raw, v_szsz_raw] = load_intralayer_series(fullfile(data_dir, files.szsz));
[d_smsp_raw, v_smsp_raw] = load_intralayer_series(fullfile(data_dir, files.smsp));
[d_spsm_raw, v_spsm_raw] = load_intralayer_series(fullfile(data_dir, files.spsm));


[d_szsz, v_szsz] = aggregate_by_distance(d_szsz_raw, v_szsz_raw, Lx);
[d_smsp, v_smsp] = aggregate_by_distance(d_smsp_raw, v_smsp_raw, Lx);
[d_spsm, v_spsm] = aggregate_by_distance(d_spsm_raw, v_spsm_raw, Lx);


% ---- Intersect Δx across all three series ----
common_d = intersect(intersect(d_szsz, d_smsp), d_spsm);
common_d = common_d(common_d > 0 & common_d <= Lx/2);
common_d = sort(common_d);
if isempty(common_d)
    error('No common Δx values across three series up to Lx/2.');
end

% Align to common Δx
[~, loc_sz] = ismember(common_d, d_szsz);  v_sz = v_szsz(loc_sz);
[~, loc_mn] = ismember(common_d, d_smsp);  v_mn = v_smsp(loc_mn);
[~, loc_pm] = ismember(common_d, d_spsm);  v_pm = v_spsm(loc_pm);


% ---- Consistency checks ----
% 1) smsp ≈ spsm (absolute error)
abs_diff_pm_mn = abs(v_pm - v_mn);
max_abs_diff   = max(abs_diff_pm_mn);
tol_abs = 1e-14;
if max_abs_diff > tol_abs
    warning('Check smsp ≈ spsm failed: max |Δ| = %.3e (tol %.1e)', max_abs_diff, tol_abs);
else
    fprintf('Check smsp ≈ spsm passed: max |Δ| = %.3e (tol %.1e)\n', max_abs_diff, tol_abs);
end

% 2) 2*szsz ≈ smsp (relative error)
two_sz = 2 * v_sz;
eps_small = 1e-20;
rel_err = abs(two_sz - v_mn) ./ max(abs(v_mn), eps_small);
max_rel_err = max(rel_err);
tol_rel = 0.01; % 1%
if max_rel_err > tol_rel
    warning('Check 2*szsz ≈ smsp failed: max rel err = %.3f%% (tol %.2f%%)', 100*max_rel_err, 100*tol_rel);
else
    fprintf('Check 2*szsz ≈ smsp passed: max rel err = %.3f%% (tol %.2f%%)\n', 100*max_rel_err, 100*tol_rel);
end


% ---- Build final S·S series to plot and fit ----
% Average transverse parts to reduce numerical noise, then form
% ⟨S·S⟩ = ⟨S^z S^z⟩ + 1/2(⟨S^+ S^-⟩ + ⟨S^- S^+⟩)
v_pm_mn_avg = 0.5 * (v_pm + v_mn);
v_ss = v_sz + v_pm_mn_avg;


% ---- Plot ----
figure('Position', [120, 120, 800, 600]);
hold on; box on; grid on;
set(gca, 'YScale', 'log');

plot(common_d, abs(v_ss), '-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', '\langle S\\cdot S \rangle (Δy=0)');

xlim([1, Lx/2]);
xlabel('|\\Delta x|', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Spin--spin |\langle S\\cdot S \rangle| (Δy=0)', 'FontSize', 18, 'FontWeight', 'bold');
title(sprintf('Intralayer (Δy=0) spin correlations (J_K=%g, J_\\perp=%g, U=%g, L_x=%d, D=%d)', ...
      Jk, Jperp, U, Lx, D), 'FontSize', 18, 'FontWeight', 'bold');
legend('show', 'Location', 'northeast', 'FontSize', 12);

ax = gca; ax.FontSize = 14; ax.FontWeight = 'bold';


% ---- Exponential fit on selected distances (7:2:25) using |⟨S·S⟩| ----
dsel_list = 7:2:25;
mask_fit = ismember(common_d, dsel_list);
d_fit_data = common_d(mask_fit);
y_fit_data = abs(v_ss(mask_fit));

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

    fprintf('Exponential fit on Δx = 7:2:25 -> correlation length xi = %.6f\n', xi);
end


% ---- Local helper functions (must be at end of script) ----
function [dist_sorted, v_sorted] = load_intralayer_series(fname)
    if ~exist(fname, 'file')
        error('File not found: %s', fname);
    end
    raw = jsondecode(fileread(fname));
    % raw is a cell array: { { [i_ref, j_ref], [Re, Im] }, { [i_ref, j], [Re, Im] }, ... }
    ref_pair = raw{1}{1};
    ref_i = ref_pair(1);

    target_i = zeros(numel(raw), 1);
    vals     = zeros(numel(raw), 1);
    for n = 1:numel(raw)
        pair = raw{n}{1};
        target_i(n) = pair(2);
        val_ri = raw{n}{2};
        vals(n) = val_ri(1); % real part only
    end

    delta_idx = abs(target_i - ref_i);
    same_row = mod(delta_idx, 8) == 0; % intralayer, same y
    dist = delta_idx(same_row) / 8;    % integer Δx
    v    = vals(same_row);

    % Exclude Δx == 0 entirely
    nz = dist > 0;
    dist = dist(nz);
    v    = v(nz);

    % Sort by distance
    [dist_sorted, idx] = sort(dist);
    v_sorted = v(idx);
end

function [u_d, u_v] = aggregate_by_distance(dlist, vlist, Lx)
    [u_d_all, ~, ic] = unique(dlist);
    u_v_all = accumarray(ic, vlist, [], @mean);
    keep = (u_d_all <= Lx/2);
    u_d = u_d_all(keep);
    u_v = u_v_all(keep);
end


