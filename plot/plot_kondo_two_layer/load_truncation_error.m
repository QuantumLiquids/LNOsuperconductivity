
% ============================================================================
% Helper function to load truncation error for a specific D value
% ============================================================================
function te = load_truncation_error(data_dir, Jperp, D)
    % Load truncation error from 201_202_bond_last_sweep_errors_JperpX.csv files
    % Inputs:
    %   data_dir: Directory containing the truncation error files
    %   Jperp: Jperp value (integer or fractional)
    %   D: Bond dimension value
    % Output:
    %   te: Truncation error value, or 1/D if file not found
    
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
            % Read the CSV file using readtable (much simpler!)
            data = readtable(te_file_path);
            
            % Find the row with matching D value
            d_idx = find(data.Bond_Dimension == D, 1);
            
            if ~isempty(d_idx)
                % Extract the truncation error
                te = data.Last_Sweep_Truncation_Error(d_idx);
                fprintf('D=%d: Loaded truncation error = %.2e\n', D, te);
            else
                warning('No truncation error found for D=%d in %s, using 1/D approximation', D, te_filename);
                te = 1 / D;
            end
            
        catch ME
            warning('Failed to load truncation error for D=%d: %s. Using 1/D approximation', D, ME.message);
            te = 1 / D;
        end
    else
        warning('Truncation error file %s not found, using 1/D approximation', te_filename);
        te = 1 / D;
    end
end
