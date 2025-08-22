% plot_sc_corr_two_layer.m
%
% Purpose
%   Plot singlet and triplet (Sz=0) pairing correlators vs distance for
%   multiple D to check convergence in the two-layer Kondo model.
%
% Behavior
%   Documentation only; plotting logic unchanged.
% Plot Superconductivity Correlations for Two-Layer Kondo Model to check convergence vs. D
clear;
close all;

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

% --- Visualization ---
figure('Position', [100, 100, 1400, 700]);
colors = lines(numel(files)); % Generate distinct colors for each D

% --- Subplot 1: Singlet SC Correlation ---
subplot(1, 2, 1);
hold on; box on; grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');

% --- Subplot 2: Triplet SC Correlation (Sz=0) ---
subplot(1, 2, 2);
hold on; box on; grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');

% --- Loop over different D values and plot ---
for i = 1:numel(files)
    D = sorted_d_values(i);
    color = colors(i, :);
    
    % Handle both integer and fractional Jperp values in filename
    if Jperp == round(Jperp)
        % Integer Jperp (e.g., 4)
        file_postfix = sprintf('conventional_squareJk%dJperp%dU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
    else
        % Fractional Jperp (e.g., 0.1, 0.5)
        file_postfix = sprintf('conventional_squareJk%dJperp%.1fU%dLx%dD%d.json', Jk, Jperp, U, Lx, D);
    end
    
    % Load the combined data using the helper function
    % Note: Using a try-catch block in case some files are missing for a given D
    try
        [scs, sct, ref_sites, target_bonds] = load_sc_data(data_dir, '', file_postfix);
    catch ME
        warning('Could not load data for D=%d. Skipping. Error: %s', D, ME.message);
        continue;
    end
    
    % --- Process distances (same for all D) ---
    ref_x = ref_sites(1) / (4*2);
    target_x = target_bonds(:,1) / (4*2);
    distances = abs(target_x - ref_x);
    [distances_sorted, sort_idx] = sort(distances);
    
    % --- Plot Singlet Data for current D ---
    subplot(1, 2, 1);
    scs_sorted = scs(sort_idx);
    loglog(distances_sorted, abs(scs_sorted), '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', color, 'DisplayName', ['D = ', num2str(D)]);
    
    % --- Plot Triplet Data (Sz=0 component) for current D ---
    subplot(1, 2, 2);
    triplet_sz0_sorted = sct(1, sort_idx);
    loglog(distances_sorted, abs(triplet_sz0_sorted), '-s', 'LineWidth', 2, 'MarkerSize', 6, 'Color', color, 'DisplayName', ['D = ', num2str(D)]);
end

% --- Finalize Plots ---
% Singlet plot
subplot(1, 2, 1);
xlabel('Distance |x_i - x_{ref}|', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Singlet SC Correlation', 'FontSize', 18, 'FontWeight', 'bold');
title('Singlet Pairing Convergence', 'FontSize', 20, 'FontWeight', 'bold');
legend('show', 'Location', 'northeast', 'FontSize', 14);
ax = gca;
ax.FontSize = 14;
ax.FontWeight = 'bold';
hold off;

% Triplet plot
subplot(1, 2, 2);
xlabel('Distance |x_i - x_{ref}|', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Triplet SC Correlation (S_z=0)', 'FontSize', 18, 'FontWeight', 'bold');
title('Triplet Pairing (S_z=0) Convergence', 'FontSize', 20, 'FontWeight', 'bold');
legend('show', 'Location', 'northeast', 'FontSize', 14);
ax = gca;
ax.FontSize = 14;
ax.FontWeight = 'bold';
hold off;

sgtitle(sprintf('SC Correlation Convergence (J_K=%d, J_{perp}=%g, U=%d, Lx=%d)', Jk, Jperp, U, Lx), 'FontSize', 24, 'FontWeight', 'bold');
