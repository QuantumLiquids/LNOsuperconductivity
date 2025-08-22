function s = format_number(val, fmt)
% FORMAT_NUMBER Safe numeric to string with optional sprintf format.
    if nargin < 2 || isempty(fmt)
        if isnumeric(val) && isscalar(val)
            s = num2str(val);
        else
            error('format_number:InvalidInput','Value must be scalar numeric without format');
        end
    else
        s = sprintf(fmt, val);
    end
end


