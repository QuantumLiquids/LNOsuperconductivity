
% Jh = 2, tilted 2-leg
colororder("gem");
C = colororder;

% Define marker size
my_marker_size = 100;
my_marker_size_square = 80;

% star, (pi/2, pi/2) state
U = [30];
t_prime = [0.3 ];
h1 = scatter(U, t_prime, 120, "filled", 'pentagram'); hold on;


% (0, pi) state
U = [1:9, 0, 10];
t_prime = [0.3*ones(1,9), 0.6,1];
h2 = scatter(U, t_prime, my_marker_size, "filled"); hold on;


% % PDW phase
% U = [3, 4, 6, 8, 2, 4, 6, 8, 6, 2,4,6 8];
% t_prime = [1/32, 1/32, 1/32, 1/32, 1/8, 1/8, 1/8, 1/8, 1/4,1/2,1/2,1/2,1/2];
% h3 = scatter(U, t_prime,my_marker_size, "filled"); hold on;
% 
% U = [8, 8,4,  2,2 , 1,1,1];
% nf = [0.6141172698111564, 0.7857378284644683,  ...
%     0.6477169003514064, ...
%     0.8832484910611048,0.7038300868950937,...
%     0.7598409960094165, 0.7029184804675991, 0.5253647279837241];
% t_prime = 1-nf;
% scatter(U, t_prime,my_marker_size_square, "filled", 's', 'MarkerFaceColor', h3.CData); hold on;
% 
% % uniform-SC phase
% U = [0, 0, 0.3, 1,0.1];
% t_prime = [1/32, 1/8, 1/8, 1/8,1/2];
% h4 = scatter(U, t_prime,my_marker_size, "filled");
% 
% U = [0.3,0.3,0.3];
% nf = [0.5648059912452188,  0.7455, 0.7819];
% t_prime = 1-nf ;
% scatter(U, t_prime, my_marker_size_square,"filled", 's', 'MarkerFaceColor', h4.CData); hold on;


%phase boundary line
%left - right line
x = [11,13, 16, 24];
y = [0,0.2, 0.4, 1];
y_fine = linspace(min(y), max(y), 100);

% Perform cubic spline interpolation
x_fine = spline(y, x, y_fine);

% Plot the smooth curve
plot(x_fine, y_fine, 'k-');

% 
% %PDW - gap line
% x_start = spline(y, x, 0.015);
% plot([x_start,8],[0.015, 0.015],'k-' );
% %s-wave -CDW line
% 
% x_target = 1.3;
% y_target = interp1(x_fine, y_fine, x_target, 'linear');
% plot([0, x_target],[0.015, y_target],'k-' );

set(gca, 'fontsize', 20);
set(gca, 'linewidth', 1.5);
set(get(gca, 'Children'), 'linewidth', 2);
xlabel('$U$', 'Interpreter', 'latex');
ylabel("$t'$", 'Interpreter', 'latex');
set(get(gca, 'XLabel'), 'FontSize', 20);
set(get(gca, 'YLabel'), 'FontSize', 20);
box on;