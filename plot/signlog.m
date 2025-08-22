% signlog.m
%
% Purpose
%   Compute sign(x) * |log(|x|)| elementwise, preserving the sign of x.
%   Useful for symmetric log-like scaling without zero/negative issues.
%
% Behavior
%   Documentation only; computation unchanged.
function [signlogx] = signlog(x)
signx= sign(x);
signlogx = signx.* abs(log(abs(x)));

end

