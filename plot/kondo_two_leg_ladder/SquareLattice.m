% SquareLattice.m (two-leg)
%
% Purpose
%   Utility class for 2D square lattice axes labels, bonds and colormap plots
%   used by two-leg scripts. Geometry is standard grid without tilt.
%
% Behavior
%   Documentation only; class logic unchanged.
classdef SquareLattice
    properties
        Ly % Number of sites in the y-direction
        Lx % Number of sites in the x-direction
        a % lattice constant 
    end

    methods
        function obj = SquareLattice(Ly, Lx)
            obj.Ly = Ly;
            obj.Lx = Lx;
            obj.a = 1; 
        end

        function [x_coord, y_coord] = indexToCoord(obj, site_idx)
            % Transform site index to square lattice coordinates
            % site index counts from 0
            x = fix(site_idx / obj.Ly) + 1;
            y = mod(site_idx, obj.Ly) + 1;
            x_coord = x * obj.a;
            y_coord = y * obj.a;
        end

        function site_idx = coordToIndex(obj, x, y)
            site_idx = (x / obj.a - 1) * obj.Ly + (y / obj.a - 1);
        end

        function drawXAxisLabels(obj, x_start, x_end, fontName, fontSize)
            % Draw x-axis labels below the lattice
            hold on;
            y_bottom = 0.5; % Position below the lattice
            for x = (x_start+1):(x_end+1)
                text(x, y_bottom, num2str(x-1), ...  % Convert to 0-based index
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'top', ...
                    'FontName', fontName, ...
                    'FontSize', fontSize, ...
                    'Color', 'k');
            end
            hold off;
        end
        
        function drawBonds(obj, bond_color, bond_width)
            hold on;
            for x = 1:obj.Lx
                for y = 1:obj.Ly
                    x_coord = x;
                    y_coord = y * obj.a;
                    if x < obj.Lx 
                        line([x_coord, x_coord + obj.a], [y_coord, y_coord], 'Color', bond_color, 'LineWidth', bond_width);
                    end
                    if y < obj.Ly
                        x_vert = x_coord;
                        y_vert = y_coord + obj.a;
                        line([x_coord, x_vert], [y_coord, y_vert], 'Color', bond_color, 'LineWidth', bond_width);
                    end
                end
            end
            hold off;
        end

        function drawOnsiteDataByColorMap(obj, data, colormap_name)
            % Create a 2D matrix for the lattice data
            lattice_data = NaN(obj.Ly, obj.Lx); % Initialize with NaN for empty sites
            for j = 1:size(data, 1)
                site_idx = data(j, 1);
                value = data(j, 2);
                [x, y] = obj.indexToCoord(site_idx);
                lattice_data(y, x) = value; % Note: imagesc expects (row, column), so (y, x)
            end

            % Plot the data using imagesc
            imagesc([1 obj.Lx], [1 obj.Ly], lattice_data);
            set(gca, 'YDir', 'normal'); % Set y-axis to increase upwards

            % Set colormap and color axis symmetrically around zero
            colormap(colormap_name);
            values = data(:, 2);
            % max_value = (max(values));
            % min_value = min(values);
            % caxis([-max_value, max_value]);
            colorbar; % Add a colorbar to interpret the values
        end

        function drawPartBonds(obj, bond_color, bond_width, x_start, x_end)
            hold on;
            for x = x_start+1:x_end+1
                for y= 1:obj.Ly-1
                    line([x,x],[y,y+1],'color',bond_color,'linewidth',bond_width);  hold on;
                    if x == x_end+1
                        line([x+1,x+1],[y,y+1],'color',bond_color,'linewidth',bond_width);  hold on;
                    end
                end
                for y = 1:obj.Ly
                    line([x,x+1],[y,y],'color',bond_color,'linewidth',bond_width);  hold on;
                end
            end
            hold off;
        end


        function drawOnsiteData(obj, data, circle_scale, positive_color, negative_color)
            % Draw onsite data as circles on the lattice
            hold on;
            for j = 1:size(data, 1)
                site_idx = data(j, 1);
                value = data(j, 2);
                [x_coord, y_coord] = obj.indexToCoord(site_idx);

                % Define radius for correlation circles
                radius = sqrt(abs(value)) * circle_scale;

                % Draw circle based on value sign
                if real(value) >= 0
                    color = positive_color;
                else
                    color = negative_color;
                end

                rectangle('Position', [x_coord - radius, y_coord - radius, 2 * radius, 2 * radius], ...
                          'Curvature', [1, 1], 'FaceColor', color, 'EdgeColor', 'none');
            end
            hold off;
        end
    end
end