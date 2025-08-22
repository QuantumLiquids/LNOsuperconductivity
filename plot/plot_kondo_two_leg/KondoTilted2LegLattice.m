classdef KondoTilted2LegLattice
    properties
        N             % Total number of sites
        L             % Number of unit cells (N/4)
        a             % Lattice spacing (default=1)
        bc            % Boundary condition ('OBC' or 'PBC')
        x_positions   % X-offsets for sites within unit cell
        y_positions   % Y-offsets for sites within unit cell
    end

    methods
        function obj = KondoTilted2LegLattice(total_sites, bc)
            if mod(total_sites, 4) ~= 0
                error('Total sites must be divisible by 4');
            end
            obj.N = total_sites;
            obj.L = total_sites / 4;
            obj.a = 1;
            obj.bc = bc;
            obj.x_positions = [0, 0.3, 1, 1.3]; 
            obj.y_positions = [0, -0.3, -1, -1.3];
        end

        function is_extended = isExtendedSite(~, idx)
            % Extended sites have mod(idx,4) == 0 or 2
            is_extended = (mod(idx, 4) == 0) || (mod(idx, 4) == 2);
        end


        function [x, y] = indexToCoord(obj, idx)
            unit_cell = floor(idx / 4);
            site_type = mod(idx, 4);
            base_x = floor(unit_cell/2) * obj.a;
            base_y = ceil(unit_cell/2) * obj.a;
            x = base_x + obj.x_positions(site_type + 1);
            y = base_y + obj.y_positions(site_type + 1);
        end

        function drawLattice(obj, bond_width, show_localized_sites)
            hold on;
            
            % Draw sites
            % for idx = 0:obj.N-1
            %     [x, y] = obj.indexToCoord(idx);
            %     plot(x, y, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
            % end
            
            % Draw intra-leg bonds (t)
            for uc = 0:obj.L-2
                % Left-upper leg
                [x1, y1] = obj.indexToCoord(4*uc);
                [x2, y2] = obj.indexToCoord(4*(uc+1));
                line([x1, x2], [y1, y2], 'Color', 'k', 'LineWidth', bond_width);
                
                % Right-lower leg
                [x1, y1] = obj.indexToCoord(4*uc + 2);
                [x2, y2] = obj.indexToCoord(4*(uc+1) + 2);
                line([x1, x2], [y1, y2], 'Color', 'k', 'LineWidth', bond_width);
            end
            
            % Draw inter-leg bonds (t2) - OBC specific
            for uc = 0:obj.L-2
                % Only even-to-even connections (0-6, 6-8, etc.)
                if mod(uc, 2) == 0  % Even unit cells
                    [x1, y1] = obj.indexToCoord(4*uc);
                    [x2, y2] = obj.indexToCoord(4*uc + 6);
                    line([x1, x2], [y1, y2], 'Color', 'k', 'LineWidth', bond_width, 'LineStyle', '--');
              
                    [x1, y1] = obj.indexToCoord(4*uc);
                    [x2, y2] = obj.indexToCoord(4*(uc-1) + 2);
                    line([x1, x2], [y1, y2], 'Color', 'k', 'LineWidth', bond_width, 'LineStyle', '--');
              
                end

           
            end
            
            % Additional PBC connections (if any)
            if strcmpi(obj.bc, 'PBC')
                for uc = 1:obj.L-2
                    if mod(uc, 2) == 1  % Odd unit cells
                        [x1, y1] = obj.indexToCoord(4*uc);
                        [x2, y2] = obj.indexToCoord(4*uc + 6);
                        line([x1, x2], [y1, y2], 'Color', 'b', 'LineWidth', bond_width, 'LineStyle', '--');
                    end
                end
            end

            if(show_localized_sites)
                % Draw Kondo couplings
                for uc = 0:obj.L-1
                    % Left-upper leg
                    [x1, y1] = obj.indexToCoord(4*uc);
                    [x2, y2] = obj.indexToCoord(4*uc + 1);
                    line([x1, x2], [y1, y2], 'Color', 'r', 'LineWidth', bond_width);

                    % Right-lower leg
                    [x1, y1] = obj.indexToCoord(4*uc + 2);
                    [x2, y2] = obj.indexToCoord(4*uc + 3);
                    line([x1, x2], [y1, y2], 'Color', 'r', 'LineWidth', bond_width);
                end
            end


            axis equal;
            box on;
            hold off;
        end

        function drawCorrelations(obj, corr_data, ref_idx, max_width, pos_color, neg_color)
            [ref_x, ref_y] = obj.indexToCoord(ref_idx);
            
            hold on;
            % Draw reference site
            plot(ref_x, ref_y, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k');
            
            for i = 1:size(corr_data, 1)
                idx = corr_data(i, 1);
                value = corr_data(i, 2);
                [x, y] = obj.indexToCoord(idx);
                
                % Skip self-correlation
                if idx == ref_idx, continue; end
                
                width = min(max_width, abs(value)*max_width*3);  % Scale for visibility
                if width < 0.05  % Skip negligible correlations
                    continue
                end
                
                if value >= 0
                    color = pos_color;
                else
                    color = neg_color;
                end
                
                line([ref_x, x], [ref_y, y], 'Color', color, 'LineWidth', width);
            end
            hold off;
        end
    end
end