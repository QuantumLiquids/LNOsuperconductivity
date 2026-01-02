function plot_singlet_sc_corr_extrapolation(extrapolation_order)
% plot_singlet_sc_corr_extrapolation.m
%
% Purpose
%   Extrapolate singlet SC correlators to TE→0 for a single J_perp and plot.
%   extrapolation_order: 1=linear, 2=quadratic vs truncation error.
%
% DMRG lattice mapping (two-layer, two-orbital, two-leg):
%   - 8 sites per physical x-position: 2 legs × 2 layers × 2 dof.
%   - For interlayer pairing: use first endpoint indices only
%     (ref_sites(1), target_bonds(:,1)). A bond has delta y = 0 iff
%     (target_i - ref_i) is a multiple of 8. Integer x-distance is
%     |target_i - ref_i| / 8. Only these bonds are used in analysis.
%
% Behavior
%   Documentation only; analysis and plotting unchanged.
% Plot singlet pairing correlations for Two-Layer Kondo Model with extrapolation
% extrapolation_order: 1 for linear, 2 for 2nd order polynomial
% Uses actual truncation errors if available; falls back to 1/D otherwise

if nargin < 1
    extrapolation_order = 2; % Default to linear extrapolation
end

clc; close all;

% --- Fixed Physical Parameters ---
Lx = 50;
Jk = -4;
Jperp = 4;
U = 18;

% --- Find all data files for different D ---
data_dir = '../../data/';
% Use a base file (scs_diag_a) to find all available D values
% Handle both integer and fractional Jperp values
if Jperp == round(Jperp)
    % Integer Jperp (e.g., 4)
    base_pattern = sprintf('scs_aconventional_squareJk%dJperp%dU%dLx%dD*.json', Jk, Jperp, U, Lx);
else
    % Fractional Jperp (e.g., 0.1, 0.5)
    base_pattern = sprintf('scs_aconventional_squareJk%dJperp%.1fU%dLx%dD*.json', Jk, Jperp, U, Lx);
end
files = dir(fullfile(data_dir, base_pattern));

if isempty(files)
    error('No data files found for the specified physical parameters. Check path and parameters.');
end

% Extract D values from filenames and sort them
d_values = zeros(numel(files), 1);
for i = 1:numel(files)
    s = files(i).name;
    % Use regexp to robustly extract the number after 'D'
    d_str = regexp(s, 'D(\d+)\.json', 'tokens');
    if ~isempty(d_str)
        d_values(i) = str2double(d_str{1}{1});
    end
end
[sorted_d_values, sort_idx_d] = sort(d_values);
files = files(sort_idx_d); % Sort the file list according to D

% Initialize storage for correlations and truncation errors
singlet_correlations_by_d = cell(numel(files), 1);
truncation_errors = zeros(numel(files), 1); % Will be filled with actual truncation errors

% --- Load all truncation error data globally ---
fprintf('Loading truncation error data for Jperp=%.1f...\n', Jperp);
all_truncation_errors = load_all_truncation_errors(data_dir, Jperp);
if ~isempty(all_truncation_errors)
    fprintf('Successfully loaded truncation errors for %d D values\n', height(all_truncation_errors));
else
    warning('No truncation error data found, will use 1/D approximation');
end

% --- Load and process data for each D value ---
for i = 1:numel(files)
    D = sorted_d_values(i);
    
    % Handle both integer and fractional Jperp values in filename
    if Jperp == round(Jperp)
        % Integer Jperp (e.g., 4)
        file_postfix = sprintf('conventional_squareJk%dJperp%dU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
    else
        % Fractional Jperp (e.g., 0.1, 0.5)
        file_postfix = sprintf('conventional_squareJk%dJperp%.1fU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
    end
    
    % Load the combined data using the helper function
    try
        [scs, ~, ref_sites, target_bonds] = load_sc_data(data_dir, '', file_postfix);
        
        % --- Process distances (delta y = 0 only; integer x using 8 sites/x) ---
        ref_i = ref_sites(1);
        target_i = target_bonds(:,1);
        delta_idx = abs(target_i - ref_i);
        same_row_idx = mod(delta_idx, 8) == 0;
        distances_int = delta_idx(same_row_idx) / 8;
        [distances_sorted, sort_idx] = sort(distances_int);
        
        % Filter distances <= Lx/2 and > 0 to remove boundary effects and log(0)
        max_distance = Lx/2;
        valid_idx = distances_sorted <= max_distance & distances_sorted > 0;
        distances_filtered = distances_sorted(valid_idx);
        scs_selected = scs(same_row_idx);
        scs_filtered = scs_selected(sort_idx);
        scs_filtered = scs_filtered(valid_idx);
        
        % Store singlet correlations for this D value
        singlet_correlations_by_d{i} = struct('distances', distances_filtered, 'correlations', scs_filtered);
        
        % --- Get truncation error from global data ---
        if ~isempty(all_truncation_errors)
            % Find the truncation error for this D value
            d_idx = find(all_truncation_errors.Bond_Dimension == D, 1);
            if ~isempty(d_idx)
                truncation_errors(i) = all_truncation_errors.Last_Sweep_Truncation_Error(d_idx);
                fprintf('D=%d: Found truncation error = %.2e\n', D, truncation_errors(i));
            else
                warning('No truncation error found for D=%d in global data, using 1/D approximation', D);
                truncation_errors(i) = 1 / D;
            end
        else
            % Fallback to 1/D approximation if no global data
            truncation_errors(i) = 1 / D;
        end
        
    catch ME
        warning('Could not load data for D=%d. Skipping. Error: %s', D, ME.message);
        continue;
    end
end

% --- Perform extrapolation for each distance ---
% Find common distances across all D values
common_distances = singlet_correlations_by_d{1}.distances;
for i = 2:numel(singlet_correlations_by_d)
    if ~isempty(singlet_correlations_by_d{i})
        common_distances = intersect(common_distances, singlet_correlations_by_d{i}.distances);
    end
end

% Initialize extrapolated correlations
extrapolated_correlations = zeros(size(common_distances));

% For each distance, perform extrapolation
for d_idx = 1:length(common_distances)
    distance = common_distances(d_idx);
    
    % Collect correlations at this distance for all D values
    corr_at_distance = [];
    te_at_distance = [];
    
    for i = 1:numel(singlet_correlations_by_d)
        if ~isempty(singlet_correlations_by_d{i})
            % Find the index of this distance in the current D data
            dist_idx = find(singlet_correlations_by_d{i}.distances == distance, 1);
            if ~isempty(dist_idx)
                corr_at_distance = [corr_at_distance, singlet_correlations_by_d{i}.correlations(dist_idx)];
                te_at_distance = [te_at_distance, truncation_errors(i)];
            end
        end
    end
    
    % Perform extrapolation
    if length(corr_at_distance) >= 2
        if extrapolation_order == 1
            % Linear extrapolation
            p = polyfit(te_at_distance, corr_at_distance, 1);
            extrapolated_correlations(d_idx) = polyval(p, 0); % Extrapolate to TE = 0
        elseif extrapolation_order == 2
            % 2nd order polynomial extrapolation
            p = polyfit(te_at_distance, corr_at_distance, 2);
            extrapolated_correlations(d_idx) = polyval(p, 0); % Extrapolate to TE = 0
        end
    end
end

% --- Plotting ---
figure('Position', [100, 100, 1200, 500]);

% Define order string for titles
if extrapolation_order == 1
    order_str = '1st';
else
    order_str = '2nd';
end

% Subplot 1: Raw correlations for different D values
subplot(1, 2, 1);
hold on; box on; grid on;
set(gca, 'XScale', 'log', 'YScale', 'log', 'FontName', 'Arial');

colors = lines(numel(files));
for i = 1:numel(singlet_correlations_by_d)
    if ~isempty(singlet_correlations_by_d{i})
        D = sorted_d_values(i);
        color = colors(i, :);
        loglog(singlet_correlations_by_d{i}.distances, abs(singlet_correlations_by_d{i}.correlations), ...
               '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', color, 'DisplayName', ['D = ', num2str(D)]);
    end
end

xlabel('Distance |x_i - x_{ref}|', 'FontSize', 14);
ylabel('|Singlet SC Correlation|', 'FontSize', 14);
title('Raw Singlet Correlations vs D', 'FontSize', 16);
legend('show', 'Location', 'northeast');
set(gca, 'FontSize', 12);

% Subplot 2: Extrapolated correlations with power law fitting
subplot(1, 2, 2);
set(gca, 'FontName', 'Arial');
loglog(common_distances, abs(extrapolated_correlations), 'o-', 'Color', [0.2 0.2 0.7], 'LineWidth', 3, 'MarkerSize', 10, 'DisplayName', 'Extrapolated (TE → 0)');

% Power law fitting: |C(r)| = A * r^(-alpha)
% Take log: log|C(r)| = log(A) - alpha * log(r)
% Use only profile distances r in {3,5,7,...,25}
profile_r = 3:2:25;
valid_idx = ~isnan(extrapolated_correlations) & (extrapolated_correlations ~= 0) & ismember(common_distances, profile_r);
if sum(valid_idx) >= 2
    log_distances = log(common_distances(valid_idx));
    log_correlations = log(abs(extrapolated_correlations(valid_idx)));
    
    % Linear fit in log-log space
    p = polyfit(log_distances, log_correlations, 1);
    K_sc = -p(1);  % Luttinger parameter for superconducting correlations
    log_A = p(2);   % Intercept
    A = exp(log_A); % Amplitude
    
    % Generate fitted curve extended to axis right limit
    ax2 = gca; r_min = max(min(common_distances(valid_idx)), ax2.XLim(1));
    r_max = max([ax2.XLim(2), max(common_distances)]);
    fit_distances = logspace(log10(r_min), log10(r_max), 200);
    fit_correlations = A * fit_distances.^(-K_sc);
    
    % Plot the fitted curve
    hold on;
    loglog(fit_distances, fit_correlations, 'b--', 'LineWidth', 2, 'DisplayName', sprintf('Power law fit: %.3f/r^{K_{sc}=%.3f}', A, K_sc));
    hold off;
    
    % Add fitting parameters to title
    title_str = sprintf('Extrapolated Correlations (%s order)\nPower Law: %.3f/r^{K_{sc}=%.3f}', order_str, A, K_sc);
else
    title_str = sprintf('Extrapolated Correlations (%s order)', order_str);
end

xlabel('Distance |x_i - x_{ref}|', 'FontSize', 14, 'FontName', 'Arial');
ylabel('|Singlet SC Correlation|', 'FontSize', 14, 'FontName', 'Arial');
title(title_str, 'FontSize', 16, 'FontName', 'Arial');
grid on; box on;
set(gca, 'FontSize', 12);
legend('show', 'Location', 'southwest', 'FontName', 'Arial');

sgtitle(sprintf('Singlet Pairing Extrapolation (J_K=%d, J_{perp}=%g, U=%d, Lx=%d, %s order)', ...
        Jk, Jperp, U, Lx, order_str), 'FontSize', 18, 'FontName', 'Arial');

% Save figure
filename_base = sprintf('singlet_sc_corr_extrapolation_Jk%dJperp%dU%dLx%d_%sorder', ...
                       Jk, Jperp, U, Lx, order_str);
saveas(gcf, [filename_base, '.png']);
saveas(gcf, [filename_base, '.pdf']);

% Print summary statistics
fprintf('\n=== Singlet Pairing Correlation Extrapolation Analysis ===\n');
fprintf('Physical parameters: J_K=%d, J_perp=%g, U=%d, Lx=%d\n', Jk, Jperp, U, Lx);
fprintf('Extrapolation order: %d\n', extrapolation_order);
fprintf('D values used: [%s]\n', num2str(sorted_d_values));
fprintf('Truncation errors: [%s]\n', num2str(truncation_errors, 4));
fprintf('Note: Using actual truncation errors from 201_202_bond_last_sweep_errors_Jperp*.csv files\n');
fprintf('Maximum distance analyzed: %.1f (Lx/2)\n', Lx/2);
fprintf('Number of common distances: %d\n', length(common_distances));
fprintf('Extrapolated correlation range: [%.6f, %.6f]\n', ...
        min(extrapolated_correlations), max(extrapolated_correlations));

% Print power law fitting results
valid_idx = ~isnan(extrapolated_correlations) & (extrapolated_correlations ~= 0);
if sum(valid_idx) >= 2
    log_distances = log(common_distances(valid_idx));
    log_correlations = log(abs(extrapolated_correlations(valid_idx)));
    
    % Linear fit in log-log space
    p = polyfit(log_distances, log_correlations, 1);
    log_A = p(2);   % Intercept
    A = exp(log_A); % Amplitude
    
    fprintf('\n=== Power Law Fitting Results ===\n');
    fprintf('Fitting function: |C(r)| = %.6f * r^(-K_{sc}=%.6f)\n', A, K_sc);
    fprintf('Luttinger parameter (K_{sc}): %.6f\n', K_sc);
    fprintf('Amplitude (A): %.6f\n', A);
    
    % Calculate R-squared for goodness of fit
    fit_correlations = A * common_distances(valid_idx).^(-K_sc);
    residuals = log_correlations - log(fit_correlations);
    ss_res = sum(residuals.^2);
    ss_tot = sum((log_correlations - mean(log_correlations)).^2);
    r_squared = 1 - (ss_res / ss_tot);
    fprintf('R-squared: %.6f\n', r_squared);
end

end

% ============================================================================
% Helper function to load all truncation errors for a given Jperp value
% ============================================================================
function all_te = load_all_truncation_errors(data_dir, Jperp)
    % Load all truncation errors from CSV file for a given Jperp value
    % Inputs:
    %   data_dir: Directory containing the truncation error files
    %   Jperp: Jperp value (integer or fractional)
    % Output:
    %   all_te: Table with Bond_Dimension and Last_Sweep_Truncation_Error columns
    
    % Handle both integer and fractional Jperp values in filename
    if Jperp == round(Jperp)
        % Integer Jperp (e.g., 4)
        te_filename = sprintf('201_202_bond_last_sweep_errors_Jperp%d.csv', Jperp);
    else
        % Fractional Jperp (e.g., 0.1, 0.5)
        te_filename = sprintf('201_202_bond_last_sweep_errors_Jperp%.1f.csv', Jperp);
    end
    te_file_path = fullfile(data_dir, te_filename);
    
    if exist(te_file_path, 'file')
        try
            % Read the CSV file using readtable
            all_te = readtable(te_file_path);
            fprintf('Loaded truncation errors from: %s\n', te_filename);
        catch ME
            warning('Failed to load truncation error file %s: %s', te_filename, ME.message);
            all_te = [];
        end
    else
        warning('Truncation error file %s not found', te_filename);
        all_te = [];
    end
end
