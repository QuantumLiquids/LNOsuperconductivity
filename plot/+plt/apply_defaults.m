function apply_defaults(target)
% APPLY_DEFAULTS Set consistent plotting defaults on a figure or axes.
%
% Usage
%   apply_defaults()             % apply to current axes (gca)
%   apply_defaults(ax)           % apply to the provided axes handle
%   apply_defaults(fig)          % apply to the provided figure handle
%
% Notes
% - This is an opt-in utility; existing scripts are not modified to call it.
% - It does not return anything and does not change plotting data.
% - Call it at the beginning of a plotting script to unify appearance.

    if nargin == 0 || isempty(target)
        target = gca; %#ok<GCA>
    end

    if ishghandle(target, 'figure')
        ax = get(target, 'CurrentAxes');
        if isempty(ax)
            ax = axes('Parent', target); %#ok<LAXES>
        end
    elseif ishghandle(target, 'axes')
        ax = target;
    else
        error('apply_defaults:InvalidHandle', 'Target must be a figure or axes handle.');
    end

    % Font and line aesthetics
    set(ax, 'FontName', 'Arial');
    set(ax, 'FontSize', 18);
    set(ax, 'LineWidth', 1.5);
    box(ax, 'on');

    % Also adjust children line widths if any exist (no try/catch)
    ch = get(ax, 'Children');
    for i = 1:numel(ch)
        if isprop(ch(i), 'LineWidth')
            set(ch(i), 'LineWidth', 1.5);
        end
    end
end


