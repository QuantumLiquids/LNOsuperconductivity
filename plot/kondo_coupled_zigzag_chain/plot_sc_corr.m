% plot_kondo_two_leg/plot_sc_corr.m
%
% Purpose
%   Load and plot superconducting correlation magnitudes vs distance for the
%   two-leg tilted Kondo lattice. Use semilog-y scaling by default.
%
% Inputs (configured in-file)
%   L, t2, Jk, U, Db  - model/truncation parameters used in file postfix
%   link_type         - 'hori' or 'diag' pairing channel
%
% Data dependencies
%   Uses ../../data/ files loaded via helper load_sc_data.m (same folder).
%
% Behavior
%   Documentation only; plotting logic is unchanged.
% clear;
% close all;

% Parameters
L  = 100;         % Number of unit cells
t2 =   1;
Jk =  -4;
U  =  18;
Db = 8000;
link_type = 'hori'; % hori, diag

FileNamePostfix = ['t2', num2str(t2), 'Jk', num2str(Jk), 'U', num2str(U), 'Lx', num2str(L),  'D', num2str(Db), '.json'];
directory = '../../data/';

[scs, sct] = load_sc_data(directory, link_type, FileNamePostfix);

% loglog(2:numel(scs)+1, abs(scs));
semilogy(2:numel(scs)+1,abs(sct));

% Set plot properties
xlabel('r', 'FontName','Arial');
ylabel("\Phi", 'FontName','Arial');
set(gca, 'FontSize', 24);
set(gca, 'LineWidth', 1.5);
set(get(gca, 'Children'), 'LineWidth', 1.5); % Set line width for all children (plots and errorbars)
set(get(gca, 'XLabel'), 'FontSize', 24);
set(get(gca, 'YLabel'), 'FontSize', 24);
box on;
% xlim([5,7]);
% hold off;
