% plot/kondo_1d_chain/plot_itinerate_spin_corr_1d_chain.m
%
% Purpose
%   Plot equal-time spin correlations S_i · S_j for the 1D Kondo chain on itinerate electrons
%   for
%   two parameter sets, using data in ../../data/. The script auto-discovers
%   available truncation bond-dimensions D from file names, records them in an
%   annotation, and uses the largest D for each parameter set.
%
% Data file patterns (any of the D suffix may be missing):
%   szszJk<JK>U<U>L100D<D>.json
%   spsmJk<JK>U<U>L100D<D>.json
%   smspJk<JK>U<U>L100D<D>.json
%
% Notes
%   - We combine as S_i · S_j = ⟨S^z_i S^z_j⟩ + ⟨S^+_i S^-_j⟩, matching usage
%     in the two-leg plotting scripts. In the present datasets smsp ≈ spsm.
%   - Marker face indicates sign: filled = positive, hollow = negative.
%   - Axes are log-log: x = distance between conduction-electron sites;
%     if indices advance by 2, we divide by 2 so the spacing is 1 per site.

clear;
close all;

% -----------------------------------------------------------------------------
% Parameter sets
params(1).Jk = -10; params(1).U = 10; % expected same-sign correlations
params(2).Jk =  -2; params(2).U =  4; % expected sign-alternating
L = 100;                                % system length used in files

% Colors per parameter set (pleasant, high-contrast)
series_colors = [
    117 112 179    % deep red
    27 158 119;   % deep blue
]/255;

% Marker appearance
marker_size = 7;          % circle size (bigger markers)
marker_edge_width = 1.5;  % thicker marker edge

data_dir = fullfile(fileparts(mfilename('fullpath')), '../../data');

% -----------------------------------------------------------------------------
% Discover files and load correlations for each parameter set
nset = numel(params);
dist_all = cell(1, nset);
corr_all = cell(1, nset);
sign_all = cell(1, nset);
available_D_note = strings(1, nset);
used_D = zeros(1, nset);

for k = 1:nset
    pk = params(k);
    base_token = sprintf('Jk%dU%dL%d', pk.Jk, pk.U, L);

    % Gather candidate files (may or may not include D)
    patt_sz = sprintf('szsz%s*.json', base_token);
    patt_pm = sprintf('spsm%s*.json', base_token);
    patt_mp = sprintf('smsp%s*.json', base_token);
    files_sz = dir(fullfile(data_dir, patt_sz));
    files_pm = dir(fullfile(data_dir, patt_pm));
    files_mp = dir(fullfile(data_dir, patt_mp)); %#ok<NASGU>

    % Parse available D values (missing D -> treat as 0)
    Ds = unique([ extract_D_values({files_sz.name}), ...
                  extract_D_values({files_pm.name}) ]);
    if isempty(Ds)
        error('No spin-correlation files found for %s under %s', base_token, data_dir);
    end
    used_D(k) = max(Ds);
    % Record a compact note of available Ds
    available_D_note(k) = sprintf('Jk=%d, U=%d: available D = {%s}, use D = %d', ...
        pk.Jk, pk.U, strjoin(string(sort(Ds)), ', '), used_D(k));
    fprintf('[SpinCorr 1D] %s\n', available_D_note(k));

    % Resolve filenames for the chosen D (or no-D if D==0)
    if used_D(k) > 0
        post = sprintf('D%d.json', used_D(k));
        f_sz = pick_by_suffix(files_sz, post, sprintf('%s.json', base_token));
        f_pm = pick_by_suffix(files_pm, post, sprintf('%s.json', base_token));
    else
        f_sz = pick_by_suffix(files_sz, sprintf('%s.json', base_token), '');
        f_pm = pick_by_suffix(files_pm, sprintf('%s.json', base_token), '');
    end

    % Load JSON and build correlation array (robust to Nx2-cell or cell-of-cell)
    SpinCorrDataZZ = jsondecode(fileread(fullfile(data_dir, f_sz)));
    SpinCorrDataPM = jsondecode(fileread(fullfile(data_dir, f_pm)));
    [ref_site_idx, target_site_idx, SpinCorr] = parse_spin_corr(SpinCorrDataZZ, SpinCorrDataPM);

    % Convert to distances along the conduction-electron chain
    raw_dist = target_site_idx - ref_site_idx;
    if all(mod(raw_dist, 2) == 0)
        dist = raw_dist / 2; % conduction sites interleaved with localized sites
    else
        dist = raw_dist;
    end

    % Store per set
    dist_all{k} = dist(:)';
    corr_all{k} = abs(SpinCorr(:)');
    sign_all{k} = sign(SpinCorr(:)');
end

% -----------------------------------------------------------------------------
% Plot |S_i · S_j| vs distance (log-log), filled = positive, hollow = negative
figure;
hold on;
set(gcf, 'Color', 'w');
set(gca, 'LineWidth', 2.0); % thicker axes frame

% keep line handles for legend alignment
h_series = gobjects(1, nset);
for k = 1:nset
    x = dist_all{k};
    y = corr_all{k};
    s = sign_all{k};
    % Truncate to x < 60
    mplot = (x > 0) & (x < 60);
    x = x(mplot); y = y(mplot); s = s(mplot);
    col = series_colors(k, :);

    % Draw connecting line for magnitude
    h_series(k) = loglog(x, y, '-', 'Color', col, 'LineWidth', 2.0);

    % Positive points: filled
    pos = s >= 0;
    loglog(x(pos), y(pos), 'o', 'MarkerEdgeColor', col, ...
        'MarkerFaceColor', col, 'MarkerSize', marker_size, 'LineWidth', marker_edge_width, ...
        'HandleVisibility','off');

    % Negative points: hollow
    neg = s < 0;
    loglog(x(neg), y(neg), 'o', 'MarkerEdgeColor', col, ...
        'MarkerFaceColor', 'none', 'MarkerSize', marker_size, 'LineWidth', marker_edge_width, ...
        'HandleVisibility','off');
end

% Power-law fit on the red series (Jk=-2, U=4), even sites 6:2:58
idx_red = find([params.Jk] == -2 & [params.U] == 4, 1);
if ~isempty(idx_red)
    x_all = dist_all{idx_red};
    y_all = corr_all{idx_red};
    % restrict to domain
    x_sel = 6:2:58;
    [is_mem, loc] = ismember(x_sel, x_all);
    x_fit = x_sel(is_mem);
    y_fit = y_all(loc(is_mem));
    if numel(x_fit) >= 2
        p = polyfit(log(x_fit), log(y_fit), 1); % log(y)=p1*log(x)+p2
        slope = p(1); intercept = p(2);
        alpha = -slope; % y ~ C * x^{-alpha}
        % extend dashed fit to 60 for a smoother line
        x_line = logspace(log10(min(x_fit)), log10(60), 200);
        y_line = exp(intercept) * (x_line .^ slope);
        h_fit = loglog(x_line, y_line, '--', 'Color', series_colors(idx_red,:), 'LineWidth', 2.0);
        fit_legend = sprintf('Fit (red): x^{-%.3f}', alpha);
    else
        fit_legend = '';
    end
else
    fit_legend = '';
end

box on;
% grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');
xlim([1 60]);
xlabel('\Delta x', 'Interpreter', 'tex');
ylabel('Spin correlation  F(\Delta x)', 'Interpreter', 'tex');
% Font sizes to match PhaseDiagram1D.m style
set(gca, 'FontName', 'Arial', 'FontSize', 20);    % tick labels
set(get(gca, 'XLabel'), 'FontSize', 20);
set(get(gca, 'YLabel'), 'FontSize', 20);
set(get(gca, 'Children'), 'LineWidth', 2);        % plotted objects

% Legend built from the main line handles to ensure alignment
leg_handles = h_series(:)';
leg_texts = cell(1, numel(leg_handles));
for k = 1:nset
    leg_texts{k} = sprintf('J_k = %d, U = %d', params(k).Jk, params(k).U);
end
if exist('h_fit','var') && ~isempty(fit_legend) && strlength(fit_legend) > 0
    leg_handles = [leg_handles, h_fit]; %#ok<AGROW>
    leg_texts   = [leg_texts,   {fit_legend}]; %#ok<AGROW>
end
legend(leg_handles, leg_texts, 'Location', 'southwest', 'Box','off');

% D-availability annotation removed; printed to terminal instead

% Tighten axes
% axis tight;


% -----------------------------------------------------------------------------
% Transparent vector export to figures/
try
    % Transparent export temporarily disabled per request
    % set(gcf, 'Color','none', 'InvertHardcopy','off', 'Renderer','painters');
    % set(findall(gcf, 'Type','axes'), 'Color','none');
    this_file = mfilename('fullpath');
    if isempty(this_file)
        this_dir = pwd;
    else
        this_dir = fileparts(this_file);
    end
    fig_dir = fullfile(this_dir, 'figures');
    if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

    base_name = sprintf('kondo_1d_chain_spin_corr_two_params_Jk%dU%d_and_Jk%dU%d', ...
        params(1).Jk, params(1).U, params(2).Jk, params(2).U);
    pdf_path = fullfile(fig_dir, [base_name, '.pdf']);
    eps_path = fullfile(fig_dir, [base_name, '.eps']);
    exportgraphics(gcf, pdf_path, 'ContentType','vector', 'BackgroundColor','none');
    print(gcf, '-depsc', '-painters', '-r600', eps_path);
catch ME
    warning(ME.identifier, '%s', ME.message);
end

% =============================================================================
% Helpers
function Ds = extract_D_values(names)
    % names: cellstr of file names
    if isempty(names)
        Ds = [];
        return;
    end
    names = names(~cellfun(@isempty, names));
    if isempty(names)
        Ds = [];
        return;
    end
    Ds = zeros(1, numel(names));
    for i = 1:numel(names)
        nm = names{i};
        tok = regexp(nm, 'D(\d+)\.json$', 'tokens', 'once');
        if isempty(tok)
            Ds(i) = 0; % missing D in file name
        else
            Ds(i) = str2double(tok{1});
        end
    end
end

function fname = pick_by_suffix(files, prefer_suffix, fallback_name)
    % Return the file name from files whose name ends with prefer_suffix; if not
    % found and fallback_name is non-empty, return fallback_name (must exist).
    fname = '';
    for i = 1:numel(files)
        if endsWith(files(i).name, prefer_suffix)
            fname = files(i).name; return; %#ok<*NASGU>
        end
    end
    if isempty(fname)
        if ~isempty(fallback_name)
            % Verify the fallback exists among candidates
            for i = 1:numel(files)
                if strcmp(files(i).name, fallback_name)
                    fname = files(i).name; return;
                end
            end
        end
    end
    if isempty(fname)
        % As a last resort, take the first entry if any exist
        if ~isempty(files)
            fname = files(1).name;
        else
            error('No candidate files to pick from.');
        end
    end
end

function [ref_idx, tgt_idx, corr_val] = parse_spin_corr(ZZ, PM)
    % Accepts either:
    %   1) cell array of length N with each entry a 1x2 cell: { [i j], [val, 0] }
    %   2) numeric Nx2x? array with first column indices packed in a vector
    % We standardize to vectors of length N.
    if iscell(ZZ)
        % Case 1: cell-of-cell
        N = numel(ZZ);
        ref_idx = ZZ{1}{1}(1);
        tgt_idx = zeros(1, N);
        corr_val = zeros(1, N);
        for ii = 1:N
            tgt_idx(ii) = ZZ{ii}{1}(2);
            corr_val(ii) = ZZ{ii}{2} + PM{ii}{2};
        end
        return;
    end

    % Case 2: try to interpret as plain numeric array of shape [N, 2, ...]
    % Expect columns like: [ [i j], [value, 0] ] per row when flattened
    try
        % Convert JSON struct/array to MATLAB array if needed
        A = ZZ;
        N = size(A, 1);
        % indices
        ij = squeeze(A(:,1,:));
        if size(ij,2) < 2
            error('Index column malformed');
        end
        ref_idx = ij(1,1);
        tgt_idx = ij(:,2)';
        % values
        vp = squeeze(PM(:,2,:));
        vz = squeeze(A(:,2,:));
        if size(vp,2) < 1 || size(vz,2) < 1
            error('Value column malformed');
        end
        corr_val = (vz(:,1) + vp(:,1))';
        return;
    catch
        error('Unrecognized spin-correlation JSON structure.');
    end
end


