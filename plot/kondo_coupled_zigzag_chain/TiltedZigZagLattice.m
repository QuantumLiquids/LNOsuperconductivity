% TiltedZigZagLattice.m
%
% Purpose
%   Geometry helper matching the C++ TiltedZigZagLattice while keeping the
%   legacy orientation used by the historic two-leg MATLAB plots. Supports
%   arbitrary Ly.
classdef TiltedZigZagLattice
    properties
        Ly
        Lx
        N
        bc
    end

    methods
        function obj = TiltedZigZagLattice(ly, lx, bc)
            arguments
                ly (1,1) {mustBePositive, mustBeInteger}
                lx (1,1) {mustBePositive, mustBeInteger}
                bc (1,:) char {mustBeMember(bc, {'OBC','PBC'})} = 'OBC'
            end
            obj.Ly = ly;
            obj.Lx = lx;
            obj.N = ly * lx;
            obj.bc = bc;
        end

        function idx = electronIndex(obj, y, x)
            idx = y + obj.Ly * x;
        end

        function is_extended = isExtendedSite(obj, idx)
            if idx < 0 || idx >= obj.N
                error('Index out of range: %d', idx);
            end
            is_extended = true;
        end

        function [x, y] = indexToCoord(obj, idx)
            xchain = floor(idx / obj.Ly);
            yleg = mod(idx, obj.Ly);
            base = floor(xchain / 2);
            x_phys = base + yleg;
            if mod(xchain, 2) == 0
                y_phys = base - yleg;
            else
                y_phys = base + 1 - yleg;
            end
            x = x_phys;
            y = y_phys;
        end

        function drawLattice(obj, bond_width, show_localized_sites)
            if nargin < 2 || isempty(bond_width)
                bond_width = 1.5;
            end
            if nargin < 3
                show_localized_sites = 0;
            end

            hold on;
            for x = 0:(obj.Lx - 2)
                for y = 0:(obj.Ly - 1)
                    [x1, y1] = obj.indexToCoord(obj.electronIndex(y, x));
                    [x2, y2] = obj.indexToCoord(obj.electronIndex(y, x + 1));
                    line([x1, x2], [y1, y2], 'Color', 'k', 'LineWidth', bond_width);
                end
            end

            for x = 0:(obj.Lx - 2)
                delta = (mod(x, 2) == 0) * 2 - 1;
                for y = 0:(obj.Ly - 1)
                    target = y + delta;
                    if target >= 0 && target < obj.Ly
                        [x1, y1] = obj.indexToCoord(obj.electronIndex(y, x));
                        [x2, y2] = obj.indexToCoord(obj.electronIndex(target, x + 1));
                        line([x1, x2], [y1, y2], 'Color', 'k', 'LineWidth', bond_width, 'LineStyle', '--');
                    elseif strcmpi(obj.bc, 'PBC')
                        wrapped = mod(target, obj.Ly);
                        if wrapped < 0
                            wrapped = wrapped + obj.Ly;
                        end
                        [x1, y1] = obj.indexToCoord(obj.electronIndex(y, x));
                        [x2, y2] = obj.indexToCoord(obj.electronIndex(wrapped, x + 1));
                        line([x1, x2], [y1, y2], 'Color', 'b', 'LineWidth', bond_width, 'LineStyle', '--');
                    end
                end
            end

            if show_localized_sites
                warning('Localized site drawing not implemented for generalized lattice.');
            end

            axis equal;
            box on;
            hold off;
        end
    end
end
