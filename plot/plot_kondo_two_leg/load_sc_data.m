function [scs, sct] = load_sc_data(directory,link_type,  FileNamePostfix)
% return: row-like vector: scs (singlet-pair correlation)
%         3 by N matrix: sct (triplet-pair correlation)
%         implicitly assume distance: 2,3,4,...
scs_diag_a = jsondecode(fileread([directory, 'scs_', link_type, '_a', FileNamePostfix]));
scs_diag_b = jsondecode(fileread([directory, 'scs_', link_type,'_b', FileNamePostfix]));
scs_diag_c = jsondecode(fileread([directory, 'scs_', link_type,'_c', FileNamePostfix]));
scs_diag_d = jsondecode(fileread([directory, 'scs_', link_type, '_d', FileNamePostfix]));

scs_diag_e = jsondecode(fileread([directory, 'sct_', link_type, '_e', FileNamePostfix]));
scs_diag_f = jsondecode(fileread([directory, 'sct_', link_type, '_f', FileNamePostfix]));

data_num = numel(scs_diag_a);
scs = zeros(1, data_num); % singlet pair
sct = zeros(3, data_num); % triplet pair
% real DMRG code
for i=1:data_num
    % hubbard like
    scs(i) = scs_diag_a{i}{2} ...
        + scs_diag_b{i}{2} ...
        + scs_diag_c{i}{2} ...
        + scs_diag_d{i}{2};
    sct(1, i) = scs_diag_a{i}{2} ...
        - scs_diag_b{i}{2} ...
        - scs_diag_c{i}{2} ...
        + scs_diag_d{i}{2};

    sct(2, i) = scs_diag_e{i}{2};
    sct(3, i) = scs_diag_f{i}{2};
end

end