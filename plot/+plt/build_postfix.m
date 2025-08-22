function postfix = build_postfix(params, varargin)
% BUILD_POSTFIX Build a filename postfix from a params struct and pattern.
%
% Usage
%   postfix = plt.build_postfix(params, pattern...)
%
% Pattern items (varargin):
%   - 'literal'                     -> appended as-is if not a params field
%   - 'fieldName'                   -> if it's a field of params, append its value via num2str
%   - {'fieldName', 'fmt'}          -> append sprintf(fmt, params.(fieldName))
%
% Example
%   params = struct('Ly_phy',2,'Lx_phy',20,'t',3.0,'J',1,'Jperp',3,'delta',0.3,'Hole',0,'D',8000);
%   p = plt.build_postfix(params, 'Ly_phy','x','Lx_phy','t',{'t','%.1f'}, ...
%       'J',{'J','%.1f'},'Jperp',{'Jperp','%.1f'},'delta','Hole','D','Pin.json');

    parts = cell(1, numel(varargin));
    for i = 1:numel(varargin)
        item = varargin{i};
        if ischar(item) || (isstring(item) && isscalar(item))
            key = char(item);
            if isfield(params, key)
                val = params.(key);
                parts{i} = local_format_default(val);
            else
                parts{i} = key;
            end
        elseif iscell(item) && numel(item) == 2
            key = item{1}; fmt = item{2};
            val = params.(key);
            parts{i} = sprintf(fmt, val);
        else
            error('build_postfix:InvalidPattern', 'Invalid pattern element at position %d', i);
        end
    end

    postfix = strjoin(parts, '');
end

function s = local_format_default(val)
    if isnumeric(val) && isscalar(val)
        % Default: compact numeric without trailing spaces
        s = num2str(val);
    elseif ischar(val) || (isstring(val) && isscalar(val))
        s = char(val);
    else
        error('build_postfix:UnsupportedType', 'Unsupported value type for default formatting');
    end
end


