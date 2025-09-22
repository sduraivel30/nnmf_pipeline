function results = load_nnmf_results(filepath)
% LOAD_NNMF_RESULTS Load NNMF analysis results from Python
%
% Usage:
%   results = load_nnmf_results('path/to/results.mat')
%
% Output:
%   results - Structure containing all NNMF analysis results

% Check if file exists
if ~exist(filepath, 'file')
    error('File not found: %s', filepath);
end

% Load the .mat file
fprintf('Loading NNMF results from: %s\n', filepath);
results = load(filepath);

% Display summary information
if isfield(results, 'metadata')
    meta = results.metadata;
    fprintf('\n=== NNMF Analysis Summary ===\n');
    fprintf('Story: %s\n', meta.story_name);
    fprintf('Electrodes used: %d / %d\n', meta.filtered_electrodes, meta.original_electrodes);
    fprintf('Subjects: %d\n', meta.n_subjects);
    fprintf('Preprocessing: %s\n', meta.preprocessing_method);
    
    if isfield(results, 'final_model')
        fm = results.final_model;
        fprintf('Optimal components: %d\n', fm.n_components);
        fprintf('Optimal alpha: %.4f\n', fm.alpha);
        fprintf('Variance explained: %.3f\n', fm.variance_explained);
    end
    fprintf('============================\n\n');
end

fprintf('Successfully loaded NNMF results\n');

end