function [scs, sct, ref_sites, target_bonds] = load_sc_data(directory, link_type, FileNamePostfix)
% Returns:
%   scs: 1xN vector of singlet pair correlations
%   sct: 3xN matrix of triplet pair correlations (Sz=0, Sz=+1, Sz=-1)
%   ref_sites: 1x2 vector of the reference bond [site_i, site_j]
%   target_bonds: Nx2 matrix of the target bonds

% Construct full filenames
scs_a_file = [directory, 'scs_', link_type, 'a', FileNamePostfix];
scs_b_file = [directory, 'scs_', link_type, 'b', FileNamePostfix];
scs_c_file = [directory, 'scs_', link_type, 'c', FileNamePostfix];
scs_d_file = [directory, 'scs_', link_type, 'd', FileNamePostfix];
sct_e_file = [directory, 'sct_', link_type, 'e', FileNamePostfix];
sct_f_file = [directory, 'sct_', link_type, 'f', FileNamePostfix];

% Load all necessary raw data
scs_diag_a = jsondecode(fileread(scs_a_file));
scs_diag_b = jsondecode(fileread(scs_b_file));
scs_diag_c = jsondecode(fileread(scs_c_file));
scs_diag_d = jsondecode(fileread(scs_d_file));
sct_diag_e = jsondecode(fileread(sct_e_file));
sct_diag_f = jsondecode(fileread(sct_f_file));

% --- Process Data ---
data_num = numel(scs_diag_a);
scs = zeros(1, data_num); % singlet pair
sct = zeros(3, data_num); % triplet pair (Sz=0, Sz=+1, Sz=-1)

% Extract site information from one of the files
ref_sites = [scs_diag_a{1}{1}(1), scs_diag_a{1}{1}(2)];
target_bonds = zeros(data_num, 2);

for i = 1:data_num
    % Get target bond sites
    target_bonds(i,:) = [scs_diag_a{i}{1}(3), scs_diag_a{i}{1}(4)];
    
    % Combine raw data into physical singlet and triplet correlations
    % Based on the provided formula for Hubbard-like models
    
    % Singlet Pair Correlation
    scs(i) = scs_diag_a{i}{2} ...
           + scs_diag_b{i}{2} ...
           + scs_diag_c{i}{2} ...
           + scs_diag_d{i}{2};
           
    % Triplet, S_z = 0 component
    sct(1, i) = scs_diag_a{i}{2} ...
              - scs_diag_b{i}{2} ...
              - scs_diag_c{i}{2} ...
              + scs_diag_d{i}{2};
    
    % Triplet, S_z = +1 component (|up up> pairing)
    sct(2, i) = sct_diag_e{i}{2};
    
    % Triplet, S_z = -1 component (|down down> pairing)
    sct(3, i) = sct_diag_f{i}{2};
end

end
