function plot_singlet_sc_corr_extrapolation_all()
% plot_singlet_sc_corr_extrapolation_all.m
%
% Purpose
%   Extrapolate singlet SC correlations to TE→0 for several J_perp values and
%   plot together. Uses linear if <=3 D points else quadratic vs truncation error.
%
% Behavior
%   Documentation only; analysis and plotting unchanged.
% Plot extrapolated singlet SC correlations for multiple J_perp on one figure
% J_perp set: [0.1, 0.5, 1, 4]
% For each J_perp use only D > 6000; if number of D <= 3, use linear extrapolation,
% otherwise use 2nd-order polynomial extrapolation. Extrapolation is done vs
% truncation error (last sweep), loaded globally from CSV per J_perp.

clc; close all;

% --- Fixed Physical Parameters ---
Lx = 50;
Jk = -4;
U = 18;
Jperp_list = [0.1, 0.5, 1, 4];
min_D = 6000;

% --- Data directory ---
data_dir = '../../data/';

% --- Figure setup ---
figure('Position', [100, 100, 800, 600]);
hold on; box on; grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');
colors = lines(numel(Jperp_list));
legend_entries = cell(1, numel(Jperp_list));

% Track limits across all J_perp for nice axes
all_extrapolated_distances = [];
all_extrapolated_values = [];

for jidx = 1:numel(Jperp_list)
    Jperp = Jperp_list(jidx);

    % --- Find all data files for different D for this Jperp ---
    if Jperp == round(Jperp)
        base_pattern = sprintf('scs_aconventional_squareJk%dJperp%dU%dLx%dD*.json', Jk, Jperp, U, Lx);
    else
        base_pattern = sprintf('scs_aconventional_squareJk%dJperp%.1fU%dLx%dD*.json', Jk, Jperp, U, Lx);
    end
    files = dir(fullfile(data_dir, base_pattern));
    
    if isempty(files)
        warning('No SC data files found for Jperp=%g. Skipping.', Jperp);
        continue;
    end

    % Extract D values and filter D > 6000
    d_values = zeros(numel(files), 1);
    for i = 1:numel(files)
        s = files(i).name;
        d_str = regexp(s, 'D(\d+)\.json', 'tokens');
        if ~isempty(d_str)
            d_values(i) = str2double(d_str{1}{1});
        end
    end
    valid_idx_d = d_values > min_D & d_values > 0;
    files = files(valid_idx_d);
    d_values = d_values(valid_idx_d);

    if isempty(files)
        warning('No D > %d for Jperp=%g. Skipping.', min_D, Jperp);
        continue;
    end

    % Sort by D ascending
    [sorted_d_values, sort_idx_d] = sort(d_values);
    files = files(sort_idx_d);

    % Decide extrapolation order for this Jperp
    if numel(sorted_d_values) <= 3
        extrapolation_order = 1; % linear
    else
        extrapolation_order = 2; % quadratic
    end

    % --- Load all truncation errors globally for this Jperp ---
    all_truncation_errors = load_all_truncation_errors(data_dir, Jperp);

    % --- Load SC data for each D and prepare correlations ---
    singlet_correlations_by_d = cell(numel(files), 1);
    for i = 1:numel(files)
        D = sorted_d_values(i);
        % Build file postfix
        if Jperp == round(Jperp)
            file_postfix = sprintf('conventional_squareJk%dJperp%dU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
        else
            file_postfix = sprintf('conventional_squareJk%dJperp%.1fU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
        end

        % Load combined data
        try
            [scs, ~, ref_sites, target_bonds] = load_sc_data(data_dir, '', file_postfix);
            % Distances processing
            ref_x = ref_sites(1) / (4*2);
            target_x = target_bonds(:,1) / (4*2);
            distances = abs(target_x - ref_x);
            [distances_sorted, sort_idx] = sort(distances);
            
            % Boundary effect removal: keep <= Lx/2
            max_distance = Lx/2;
            valid_idx = distances_sorted <= max_distance;
            distances_filtered = distances_sorted(valid_idx);

            scs_filtered = scs(sort_idx);
            scs_filtered = scs_filtered(valid_idx);

            singlet_correlations_by_d{i} = struct('D', D, 'distances', distances_filtered, 'correlations', scs_filtered);
        catch ME
            warning('Jperp=%g D=%d: load_sc_data failed: %s. Skipping.', Jperp, D, ME.message);
            singlet_correlations_by_d{i} = [];
        end
    end

    % Remove empties
    singlet_correlations_by_d = singlet_correlations_by_d(~cellfun(@isempty, singlet_correlations_by_d));
    if isempty(singlet_correlations_by_d)
        warning('No valid SC data for Jperp=%g after loading. Skipping.', Jperp);
        continue;
    end

    % Common distances across D for this Jperp
    common_distances = singlet_correlations_by_d{1}.distances;
    for i = 2:numel(singlet_correlations_by_d)
        common_distances = intersect(common_distances, singlet_correlations_by_d{i}.distances);
    end
    if isempty(common_distances)
        warning('No common distances for Jperp=%g. Skipping.', Jperp);
        continue;
    end

    % Extrapolate at each common distance
    extrapolated_correlations = nan(size(common_distances));

    for d_idx = 1:length(common_distances)
        distance = common_distances(d_idx);
        corr_at_distance = [];
        te_at_distance = [];

        for i = 1:numel(singlet_correlations_by_d)
            D = singlet_correlations_by_d{i}.D;
            % Find correlation at this distance
            idx = find(singlet_correlations_by_d{i}.distances == distance, 1);
            if ~isempty(idx)
                corr_at_distance(end+1) = singlet_correlations_by_d{i}.correlations(idx); %#ok<AGROW>
                % Lookup truncation error for this D from global table
                te_val = lookup_te_for_D(all_truncation_errors, D);
                if isnan(te_val) || te_val <= 0
                    te_val = 1 / D; % fallback
                end
                te_at_distance(end+1) = te_val; %#ok<AGROW>
            end
        end

        % Determine order allowed by available points
        num_pts = numel(corr_at_distance);
        if num_pts >= 2
            if extrapolation_order == 2 && num_pts >= 3
                p = polyfit(te_at_distance, corr_at_distance, 2);
                extrapolated_correlations(d_idx) = polyval(p, 0);
            else
                % Linear fallback
                p = polyfit(te_at_distance, corr_at_distance, 1);
                extrapolated_correlations(d_idx) = polyval(p, 0);
            end
        end
    end

    % Plot extrapolated results for this Jperp
    color = colors(jidx, :);
    loglog(common_distances, abs(extrapolated_correlations), '-', 'LineWidth', 2.5, 'Color', color);
    
    % Power-law fit on extrapolated data: |C(r)| = A * r^{-K_sc}
    valid_fit = ~isnan(extrapolated_correlations) & (extrapolated_correlations > 0) & (common_distances > 0);
    if any(valid_fit)
        log_r = log(common_distances(valid_fit));
        log_c = log(abs(extrapolated_correlations(valid_fit)));
        if numel(log_r) >= 2
            pfit = polyfit(log_r, log_c, 1);
            K_sc = -pfit(1);
            A_fit = exp(pfit(2));
            r_min = min(common_distances(valid_fit));
            r_max = max(common_distances(valid_fit));
            r_fit = logspace(log10(r_min), log10(r_max), 100);
            c_fit = A_fit * r_fit.^(-K_sc);
            hold on;
            loglog(r_fit, c_fit, '--', 'LineWidth', 2, 'Color', color, 'HandleVisibility', 'off');
        else
            K_sc = NaN;
        end
    else
        K_sc = NaN;
    end
    
    % Legend entry with method info and K_sc
    method_str = ternary_str(extrapolation_order == 1, 'linear', 'quadratic');
    if ~isnan(K_sc)
        legend_entries{jidx} = sprintf('J_{\\perp}=%g (%s), K_{sc}=%.3f', Jperp, method_str, K_sc);
    else
        legend_entries{jidx} = sprintf('J_{\\perp}=%g (%s)', Jperp, method_str);
    end

    % Track global ranges
    all_extrapolated_distances = [all_extrapolated_distances; common_distances(:)]; %#ok<AGROW>
    all_extrapolated_values = [all_extrapolated_values; abs(extrapolated_correlations(:))]; %#ok<AGROW>
end

% Finalize plot
xlabel('Distance |x_i - x_{ref}|', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('|Singlet SC Correlation| (TE \rightarrow 0)', 'FontSize', 16, 'FontWeight', 'bold');
title('Extrapolated Singlet SC Correlations for J_{\\perp} = 0.1, 0.5, 1, 4', 'FontSize', 18, 'FontWeight', 'bold');
legend(legend_entries, 'Location', 'southwest', 'FontSize', 12);
ax = gca; ax.FontSize = 14; ax.FontWeight = 'bold';

% Disable axes toolbar to avoid export warning
ax.Toolbar.Visible = 'off';

% Set reasonable axis limits if data exists
if ~isempty(all_extrapolated_distances) && ~isempty(all_extrapolated_values)
    xlim([min(all_extrapolated_distances(all_extrapolated_distances>0)), max(all_extrapolated_distances)]);
    y_min = min(all_extrapolated_values(all_extrapolated_values>0));
    y_max = max(all_extrapolated_values);
    ylim([y_min, y_max]);
end

% Save figure
saveas(gcf, 'singlet_sc_corr_extrapolation_all.png');
saveas(gcf, 'singlet_sc_corr_extrapolation_all.pdf');

end

% ============================= Helpers ======================================
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
            warning('Failed to read %s', te_filename);
            all_te = [];
        end
    else
        warning('Truncation error file %s not found', te_filename);
        all_te = [];
    end
end

function te_val = lookup_te_for_D(all_te, D)
    if isempty(all_te)
        te_val = NaN; return;
    end
    if any(strcmpi(all_te.Properties.VariableNames, 'Bond_Dimension')) && any(strcmpi(all_te.Properties.VariableNames, 'Last_Sweep_Truncation_Error'))
        idx = find(all_te.Bond_Dimension == D, 1);
        if ~isempty(idx)
            te_val = all_te.Last_Sweep_Truncation_Error(idx);
            return;
        end
    end
    te_val = NaN;
end

function out = ternary_str(cond, a, b)
    if cond, out = a; else, out = b; end
end
