% plot/kondo_1d_chain/plot_corr_and_phase_2panel.m
%
% Purpose
%   Produce a two-panel figure for the 1D Kondo chain:
%     - Left: spin correlation |S_i · S_j| vs Δx (log-log), with sign via
%       filled/hollow markers and a power-law fit on the SDW-like series.
%     - Right: phase diagram with FM and 2k_F-SDW regions and boundary.
%
% Styling
%   Colors use Option A (paper-friendly, colorblind-safe):
%     FM (teal)  = [27 158 119]/255
%     SDW (purple)= [117 112 179]/255
%   Font sizes: tick=20, labels=20, line widths=2.

clear; close all;

% -----------------------------------------------------------------------------
% Shared styling and colors
fm_color  = [27 158 119]/255;   % FM: deep teal
sdw_color = [117 112 179]/255;  % 2k_F-SDW: purple
series_colors = [fm_color; sdw_color]; % params order: FM-like first, SDW-like second

marker_size = 7;          % circle marker size
marker_edge_width = 1.5;  % marker outline

font_name = 'Arial';
font_size_axes = 26;
line_width_axes = 2.0;
line_width_plot = 2.0;

% -----------------------------------------------------------------------------
% Create layout
tl = tiledlayout(1, 2, 'TileSpacing','compact', 'Padding','compact');

% Left panel: phase diagram
ax1 = nexttile(tl, 1);
draw_phase_panel(ax1, fm_color, sdw_color, font_name, font_size_axes, line_width_axes);

% Right panel: correlation (swap teal/purple order for this panel)
ax2 = nexttile(tl, 2);
draw_corr_panel(ax2, series_colors([2 1],:), marker_size, marker_edge_width, ...
    font_name, font_size_axes, line_width_axes, line_width_plot);

% Export (vector)
try
    set(gcf, 'Color','w', 'Renderer','painters');
    this_file = mfilename('fullpath');
    if isempty(this_file)
        this_dir = pwd;
    else
        this_dir = fileparts(this_file);
    end
    fig_dir = fullfile(this_dir, 'figures'); if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end
    base_name = 'kondo_1d_chain_corr_and_phase_2panel';
    exportgraphics(gcf, fullfile(fig_dir, [base_name, '.pdf']), 'ContentType','vector', 'BackgroundColor','white');
    print(gcf, '-depsc', '-painters', '-r600', fullfile(fig_dir, [base_name, '.eps']));
catch ME
    warning(ME.identifier, '%s', ME.message);
end

% ==============================================================================
% Left panel implementation -----------------------------------------------------
function draw_corr_panel(ax, series_colors, marker_size, marker_edge_width, ...
                         font_name, font_size_axes, line_width_axes, line_width_plot)

    axes(ax); cla(ax);
    hold(ax, 'on'); box(ax, 'on');
    set(ax, 'LineWidth', line_width_axes);

    % Parameters and data location
    params(1).Jk = -10; params(1).U = 10; % FM-like (same sign)
    params(2).Jk =  -2; params(2).U =  4; % SDW-like (alternating)
    L = 100;
    data_dir = fullfile(fileparts(mfilename('fullpath')), '../../data');

    % Two orbital channels: itinerant d_{x^2-y^2} and localized d_{z^2}.
    % Color = phase (FM teal / SDW purple, via series_colors); shape = orbital.
    orbitals(1).prefix_zz = 'szsz';  orbitals(1).prefix_pm = 'spsm';  orbitals(1).marker = 'o'; % itinerant d_{x^2-y^2}
    orbitals(2).prefix_zz = 'lszsz'; orbitals(2).prefix_pm = 'lspsm'; orbitals(2).marker = 's'; % localized d_{z^2}

    nset  = numel(params);
    norb  = numel(orbitals);
    dist_all = cell(norb, nset);
    corr_all = cell(norb, nset);
    sign_all = cell(norb, nset);
    used_D   = zeros(norb, nset);

    for o = 1:norb
        for k = 1:nset
            pk = params(k);
            base_token = sprintf('Jk%dU%dL%d', pk.Jk, pk.U, L);
            patt_sz = sprintf('%s%s*.json', orbitals(o).prefix_zz, base_token);
            patt_pm = sprintf('%s%s*.json', orbitals(o).prefix_pm, base_token);
            files_sz = dir(fullfile(data_dir, patt_sz));
            files_pm = dir(fullfile(data_dir, patt_pm));
            Ds = unique([ extract_D_values({files_sz.name}), extract_D_values({files_pm.name}) ]);
            if isempty(Ds)
                error('No %s/%s files found for %s under %s', ...
                    orbitals(o).prefix_zz, orbitals(o).prefix_pm, base_token, data_dir);
            end
            used_D(o,k) = max(Ds);
            if used_D(o,k) > 0
                post = sprintf('D%d.json', used_D(o,k));
                f_sz = pick_by_suffix(files_sz, post, sprintf('%s.json', base_token));
                f_pm = pick_by_suffix(files_pm, post, sprintf('%s.json', base_token));
            else
                f_sz = pick_by_suffix(files_sz, sprintf('%s.json', base_token), '');
                f_pm = pick_by_suffix(files_pm, sprintf('%s.json', base_token), '');
            end

            SpinCorrDataZZ = jsondecode(fileread(fullfile(data_dir, f_sz)));
            SpinCorrDataPM = jsondecode(fileread(fullfile(data_dir, f_pm)));
            [ref_site_idx, target_site_idx, SpinCorr] = parse_spin_corr(SpinCorrDataZZ, SpinCorrDataPM);

            raw_dist = target_site_idx - ref_site_idx;
            if all(mod(raw_dist, 2) == 0); dist = raw_dist/2; else; dist = raw_dist; end
            dist_all{o,k} = dist(:)';
            corr_all{o,k} = abs(SpinCorr(:)');
            sign_all{o,k} = sign(SpinCorr(:)');
        end
    end

    % Plot: line + markers per (orbital, phase). Color = phase, shape = orbital.
    h_series = gobjects(norb, nset);
    for o = 1:norb
        mk = orbitals(o).marker;
        for k = 1:nset
            x = dist_all{o,k}; y = corr_all{o,k}; s = sign_all{o,k};
            m = (x > 0) & (x < 60); x = x(m); y = y(m); s = s(m);
            col = series_colors(k, :);

            h_series(o,k) = loglog(ax, x, y, '-', 'Color', col, 'LineWidth', line_width_plot);
            pos = s >= 0;
            loglog(ax, x(pos), y(pos), mk, 'MarkerEdgeColor', col, 'MarkerFaceColor', col, ...
                'MarkerSize', marker_size, 'LineWidth', marker_edge_width, 'HandleVisibility','off');
            neg = s < 0;
            loglog(ax, x(neg), y(neg), mk, 'MarkerEdgeColor', col, 'MarkerFaceColor', 'none', ...
                'MarkerSize', marker_size, 'LineWidth', marker_edge_width, 'HandleVisibility','off');
        end
    end

    % Power-law fit on the itinerant SDW-like series (orbital 1, params(2))
    idx_sdw = 2; o_fit = 1;
    x_all = dist_all{o_fit, idx_sdw}; y_all = corr_all{o_fit, idx_sdw};
    x_sel = 6:2:58; [is_mem, loc] = ismember(x_sel, x_all);
    x_fit = x_sel(is_mem); y_fit = y_all(loc(is_mem));
    h_fit = []; fit_legend = '';
    if numel(x_fit) >= 2
        p = polyfit(log(x_fit), log(y_fit), 1); slope = p(1); intercept = p(2);
        alpha = -slope;
        xl = logspace(log10(min(x_fit)), log10(60), 200);
        yl = exp(intercept) * (xl .^ slope);
        h_fit = loglog(ax, xl, yl, '--', 'Color', series_colors(idx_sdw,:), 'LineWidth', line_width_plot);
        fit_legend = sprintf('Fit (SDW): r^{-%.3f}', alpha);
    end

    set(ax, 'XScale','log', 'YScale','log'); xlim(ax, [1 60]);
    xlabel(ax, 'r', 'Interpreter','tex', 'FontName','Arial');
    ylabel(ax, 'Spin correlation  F(r)', 'Interpreter','tex', 'FontName','Arial');
    set(ax, 'FontName', font_name, 'FontSize', font_size_axes);

    % --- y = 1/4 reference (saturation value of localized FM correlation) ----
    y_ref = 0.25;
    plot(ax, [1 60], [y_ref y_ref], ':', 'Color', [0.35 0.35 0.35], ...
        'LineWidth', 1.5, 'HandleVisibility','off');
    text(ax, 1.15, y_ref*1.18, '1/4', 'Color', [0.35 0.35 0.35], ...
        'FontName', font_name, 'FontSize', 18, ...
        'HorizontalAlignment','left', 'VerticalAlignment','bottom');

    % --- Orbital labels with leader lines anchored on the FM curves ---------
    % FM-like params index (params(1) is the FM-like set with Jk=-10, U=10).
    fm_idx = 1;
    label_color = [0 0 0];
    label_fs    = 18;

    % Itinerant d_{x^2-y^2}: anchor at small r where the curve is well separated.
    x_anchor_it = 4;
    [~, ii] = min(abs(dist_all{1, fm_idx} - x_anchor_it));
    y_anchor_it = corr_all{1, fm_idx}(ii);
    x_label_it  = 1.6;
    y_label_it  = y_anchor_it * 0.30;
    plot(ax, [x_label_it x_anchor_it], [y_label_it y_anchor_it], '-', ...
        'Color', label_color, 'LineWidth', 1.0, 'HandleVisibility','off');
    text(ax, x_label_it, y_label_it, 'd_{x^2-y^2}', ...
        'Color', label_color, 'FontName', font_name, 'FontSize', label_fs, ...
        'HorizontalAlignment','left', 'VerticalAlignment','top');

    % Localized d_{z^2}: anchor on the flat FM plateau near y = 1/4.
    x_anchor_lo = 25;
    [~, ij] = min(abs(dist_all{2, fm_idx} - x_anchor_lo));
    y_anchor_lo = corr_all{2, fm_idx}(ij);
    x_label_lo  = 8;
    y_label_lo  = y_anchor_lo * 2.6;
    plot(ax, [x_label_lo x_anchor_lo], [y_label_lo y_anchor_lo], '-', ...
        'Color', label_color, 'LineWidth', 1.0, 'HandleVisibility','off');
    text(ax, x_label_lo, y_label_lo, 'd_{z^2}', ...
        'Color', label_color, 'FontName', font_name, 'FontSize', label_fs, ...
        'HorizontalAlignment','left', 'VerticalAlignment','bottom');

    % --- Legend: phase-only, plain colored lines without markers ------------
    phase_handles = gobjects(1, nset);
    for k = 1:nset
        phase_handles(k) = plot(ax, NaN, NaN, '-', ...
            'Color', series_colors(k,:), 'LineWidth', line_width_plot);
    end
    phase_texts = cell(1, nset);
    for k = 1:nset
        phase_texts{k} = sprintf('J_H = %dt, U = %dt', -params(k).Jk, params(k).U);
    end
    leg_handles = phase_handles;
    leg_texts   = phase_texts;
    if ~isempty(h_fit)
        leg_handles = [leg_handles, h_fit]; %#ok<AGROW>
        leg_texts   = [leg_texts, {fit_legend}]; %#ok<AGROW>
    end
    legend(ax, leg_handles, leg_texts, 'Location','southwest', ...
        'Box','off', 'FontSize', 18);
end

% ==============================================================================
% Right panel implementation ----------------------------------------------------
function draw_phase_panel(ax, fm_color, sdw_color, font_name, font_size_axes, line_width_axes)
    axes(ax); cla(ax);
    hold(ax, 'on'); box(ax, 'on');
    set(ax, 'LineWidth', line_width_axes);

    % Marker sizes
    my_marker_size = 100;

    % 2k_F-SDW typical points (triangle)
    U = [(0:2:12), 0,2,4,6, 0,2, 0];
    Jh = [0 * ones(1,7), 2 * ones(1, 4), 4 * ones(1,2), 6 * ones(1,1)];
    U_tri = U; Jh_tri = Jh;
    scatter(ax, U, Jh, 120, 'filled', '^', 'MarkerFaceColor', fm_color, 'MarkerEdgeColor','none', 'HandleVisibility','off');

    % (0, pi) state points (SDW color)
    U = [10,12, 8,10,12, (6:2:12), 4:2:12, 0,(0:2:12)];
    Jh = [4 * ones(1,2), 6 * ones(1,3), 8*ones(1,4), 10*ones(1,5), 13,15*ones(1,7)];
    U_opi = U; Jh_opi = Jh;
    scatter(ax, U, Jh, my_marker_size, 'filled', 'MarkerFaceColor', sdw_color, 'MarkerEdgeColor','none', 'HandleVisibility','off');

    % Phase boundary (cubic spline through anchor points)
    x = [0, 4, 8, 10, 12];
    y = [12.3, 9, 5, 3.5, 2.5]-2;
    y_all = [y, Jh_tri, Jh_opi];
    y_fine = linspace(min(y_all), max(y_all), 200);
    x_fine = spline(y, x, y_fine);

    x_max = 12; y_min = min([Jh_tri, Jh_opi, y_fine]) - 0.2; y_max = max([Jh_tri, Jh_opi, y_fine]) + 0.2;

    % Background fills using same hue but lighter
    y0 = median(y_fine); x_boundary_mid = spline(y, x, y0);
    sdw_left = median(U_opi) <= x_boundary_mid;
    mix = 0.5; face_alpha = 0.25;
    sdw_bg = (1-mix)*sdw_color + mix*[1 1 1];
    fm_bg  = (1-mix)*fm_color  + mix*[1 1 1];
    if sdw_left; left_fill_color = sdw_bg; right_fill_color = fm_bg; else; left_fill_color = fm_bg; right_fill_color = sdw_bg; end

    x_left  = [zeros(size(y_fine)),      fliplr(x_fine)];
    y_left  = [y_fine,                   fliplr(y_fine)];
    x_right = [x_fine, x_max*ones(size(y_fine))];
    y_right = [y_fine, fliplr(y_fine)];
    patch(ax, x_left,  y_left,  left_fill_color,  'EdgeColor','none', 'FaceAlpha',face_alpha);
    patch(ax, x_right, y_right, right_fill_color, 'EdgeColor','none', 'FaceAlpha',face_alpha);

    plot(ax, x_fine, y_fine, 'k-', 'LineWidth', 2);

    % Labels
    text(ax, 1.5*x_boundary_mid, y0/2, '2k_F-SDW', 'FontName',font_name, 'FontSize',font_size_axes, ...
        'FontWeight','bold', 'Color', fm_color, 'HorizontalAlignment','center');
    text(ax, (x_boundary_mid+x_max)/2, y0*3/2, 'FM', 'FontName',font_name, 'FontSize',font_size_axes, ...
        'FontWeight','bold', 'Color', sdw_color, 'HorizontalAlignment','center');

    xlim(ax, [0 x_max]); ylim(ax, [0 15]);
    set(ax, 'FontName', font_name, 'FontSize', font_size_axes);
    xlabel(ax, 'U/t', 'FontName',font_name);
    ylabel(ax, 'J_H/t', 'FontName',font_name);
end

% ==============================================================================
% Helpers (duplicated minimal versions)
function Ds = extract_D_values(names)
    if isempty(names); Ds = []; return; end
    names = names(~cellfun(@isempty, names)); if isempty(names); Ds = []; return; end
    Ds = zeros(1, numel(names));
    for i = 1:numel(names)
        nm = names{i}; tok = regexp(nm, 'D(\d+)\.json$', 'tokens', 'once');
        if isempty(tok); Ds(i) = 0; else; Ds(i) = str2double(tok{1}); end
    end
end

function fname = pick_by_suffix(files, prefer_suffix, fallback_name)
    fname = '';
    for i = 1:numel(files)
        if endsWith(files(i).name, prefer_suffix); fname = files(i).name; return; end
    end
    if isempty(fname) && ~isempty(fallback_name)
        for i = 1:numel(files)
            if strcmp(files(i).name, fallback_name); fname = files(i).name; return; end
        end
    end
    if isempty(fname)
        if ~isempty(files); fname = files(1).name; else; error('No candidate files to pick from.'); end
    end
end

function [ref_idx, tgt_idx, corr_val] = parse_spin_corr(ZZ, PM)
    if iscell(ZZ)
        N = numel(ZZ); ref_idx = ZZ{1}{1}(1); tgt_idx = zeros(1,N); corr_val = zeros(1,N);
        for ii = 1:N; tgt_idx(ii) = ZZ{ii}{1}(2); corr_val(ii) = ZZ{ii}{2} + PM{ii}{2}; end; return;
    end
    try
        A = ZZ; N = size(A,1); ij = squeeze(A(:,1,:));
        if size(ij,2) < 2; error('Index column malformed'); end
        ref_idx = ij(1,1); tgt_idx = ij(:,2)';
        vp = squeeze(PM(:,2,:)); vz = squeeze(A(:,2,:));
        if size(vp,2) < 1 || size(vz,2) < 1; error('Value column malformed'); end
        corr_val = (vz(:,1) + vp(:,1))';
    catch
        error('Unrecognized spin-correlation JSON structure.');
    end
end


