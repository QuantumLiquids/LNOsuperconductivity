% clear;
% close all;

% Parameters
L  = 100;         % Number of unit cells
Jk =  -4;
U  =  18;
Db = 5000;
link_type = 'vert'; % hori, diag

FileNamePostfix = ['conventional_square', 'Jk', num2str(Jk), 'U', num2str(U), 'Lx', num2str(L),  'D', num2str(Db), '.json'];
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
