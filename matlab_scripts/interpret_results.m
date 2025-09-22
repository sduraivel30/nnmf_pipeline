function analysis = interpret_results(results, varargin)
% INTERPRET_RESULTS Interpret NNMF results and generate analysis report
%
% Usage:
%   analysis = interpret_results(results)
%   analysis = interpret_results(results, 'generate_report', true)

% Parse inputs
p = inputParser;
addRequired(p, 'results', @isstruct);
addParameter(p, 'generate_report', true, @islogical);
parse(p, results, varargin{:});

analysis = struct();
analysis.timestamp = datestr(now);

fprintf('Interpreting NNMF results...\n');

if ~isfield(results, 'final_model')
    error('Results must contain final_model field');
end

% Extract basic information
fm = results.final_model;
analysis.n_components = fm.n_components;
analysis.optimal_alpha = fm.alpha;
analysis.variance_explained = fm.variance_explained;

W = fm.W;
H = fm.H;
n_electrodes = size(W, 1);
n_timepoints = size(H, 2);

fprintf('Found %d components explaining %.1f%% of variance\n', ...
        fm.n_components, fm.variance_explained * 100);

% Analyze spatial components
fprintf('Analyzing spatial components...\n');
analysis.spatial_analysis = analyze_spatial_components(W);

% Analyze temporal components  
fprintf('Analyzing temporal components...\n');
analysis.temporal_analysis = analyze_temporal_components(H, results);

% Component interactions
fprintf('Analyzing component interactions...\n');
analysis.interaction_analysis = analyze_component_interactions(W, H);

% Cross-validation analysis
if isfield(results, 'cross_validation')
    fprintf('Analyzing cross-validation results...\n');
    analysis.cv_analysis = analyze_cross_validation(results.cross_validation);
end

% Generate interpretation report
if p.Results.generate_report
    fprintf('Generating interpretation report...\n');
    report = generate_interpretation_report(analysis, results);
    analysis.report = report;
    
    % Display report
    fprintf('\n%s\n', report);
end

fprintf('Analysis complete.\n');

end

function spatial_analysis = analyze_spatial_components(W)
    spatial_analysis = struct();
    n_components = size(W, 2);
    threshold = 0.1;
    
    for i = 1:n_components
        comp_weights = W(:, i);
        
        spatial_analysis.(sprintf('component_%d', i)) = struct(...
            'max_weight', max(comp_weights), ...
            'mean_weight', mean(comp_weights), ...
            'std_weight', std(comp_weights), ...
            'sparsity', sum(comp_weights > threshold) / length(comp_weights), ...
            'peak_electrode', find(comp_weights == max(comp_weights), 1));
    end
    
    % Overall analysis
    all_sparsities = arrayfun(@(i) spatial_analysis.(sprintf('component_%d', i)).sparsity, 1:n_components);
    spatial_analysis.summary = struct(...
        'mean_sparsity', mean(all_sparsities), ...
        'electrode_participation', sum(W > threshold, 2));
end

function temporal_analysis = analyze_temporal_components(H, results)
    temporal_analysis = struct();
    n_components = size(H, 1);
    threshold = 0.1;
    
    for i = 1:n_components
        comp_activation = H(i, :);
        activation_threshold = threshold * max(comp_activation);
        active_periods = comp_activation > activation_threshold;
        
        temporal_analysis.(sprintf('component_%d', i)) = struct(...
            'peak_activation', max(comp_activation), ...
            'peak_time', find(comp_activation == max(comp_activation), 1), ...
            'activation_duration', sum(active_periods) / length(active_periods), ...
            'rms_activation', sqrt(mean(comp_activation.^2)));
    end
    
    % Component timing relationships
    peak_times = arrayfun(@(i) temporal_analysis.(sprintf('component_%d', i)).peak_time, 1:n_components);
    temporal_analysis.summary = struct(...
        'temporal_spread', max(peak_times) - min(peak_times), ...
        'mean_activation_duration', mean(arrayfun(@(i) temporal_analysis.(sprintf('component_%d', i)).activation_duration, 1:n_components)));
end

function interaction_analysis = analyze_component_interactions(W, H)
    interaction_analysis = struct();
    
    % Spatial correlations
    spatial_corr = corr(W);
    interaction_analysis.spatial_correlations = spatial_corr;
    
    % Temporal correlations
    temporal_corr = corr(H');
    interaction_analysis.temporal_correlations = temporal_corr;
    
    % Component orthogonality
    n_components = size(W, 2);
    triu_mask = triu(true(n_components), 1);
    interaction_analysis.mean_spatial_correlation = mean(spatial_corr(triu_mask));
    interaction_analysis.mean_temporal_correlation = mean(temporal_corr(triu_mask));
end

function cv_analysis = analyze_cross_validation(cv_results)
    cv_analysis = struct();
    
    cv_analysis.optimal_components = cv_results.optimal_components;
    cv_analysis.optimal_alpha = cv_results.optimal_alpha;
    
    if isfield(cv_results, 'mean_recon_errors')
        cv_analysis.min_reconstruction_error = min(cv_results.mean_recon_errors(:));
    end
    
    if isfield(cv_results, 'mean_reliability')
        cv_analysis.max_reliability = max(cv_results.mean_reliability(:));
    end
end

function report = generate_interpretation_report(analysis, results)
    report = sprintf('NNMF ANALYSIS INTERPRETATION REPORT\n');
    report = [report sprintf('Generated: %s\n', analysis.timestamp)];
    report = [report sprintf('=====================================\n\n')];
    
    % Story information
    if isfield(results, 'metadata')
        meta = results.metadata;
        report = [report sprintf('STORY INFORMATION:\n')];
        report = [report sprintf('- Name: %s\n', meta.story_name)];
        report = [report sprintf('- Electrodes: %d filtered / %d total\n', meta.filtered_electrodes, meta.original_electrodes)];
        report = [report sprintf('- Subjects: %d\n', meta.n_subjects)];
        report = [report sprintf('- Preprocessing: %s\n\n', meta.preprocessing_method)];
    end
    
    % Model performance
    report = [report sprintf('MODEL PERFORMANCE:\n')];
    report = [report sprintf('- Components: %d\n', analysis.n_components)];
    report = [report sprintf('- Regularization (alpha): %.4f\n', analysis.optimal_alpha)];
    report = [report sprintf('- Variance explained: %.1f%%\n\n', analysis.variance_explained * 100)];
    
    % Component analysis
    if isfield(analysis, 'spatial_analysis')
        sa = analysis.spatial_analysis;
        report = [report sprintf('SPATIAL ANALYSIS:\n')];
        report = [report sprintf('- Mean sparsity: %.2f\n', sa.summary.mean_sparsity)];
        
        for i = 1:analysis.n_components
            comp = sa.(sprintf('component_%d', i));
            report = [report sprintf('  Component %d: sparsity=%.2f, peak_electrode=%d\n', ...
                     i, comp.sparsity, comp.peak_electrode)];
        end
        report = [report sprintf('\n')];
    end
    
    if isfield(analysis, 'temporal_analysis')
        ta = analysis.temporal_analysis;
        report = [report sprintf('TEMPORAL ANALYSIS:\n')];
        report = [report sprintf('- Temporal spread: %.0f samples\n', ta.summary.temporal_spread)];
        report = [report sprintf('- Mean activation duration: %.1f%%\n\n', ta.summary.mean_activation_duration * 100)];
    end
    
    % Component interactions
    if isfield(analysis, 'interaction_analysis')
        ia = analysis.interaction_analysis;
        report = [report sprintf('COMPONENT INTERACTIONS:\n')];
        report = [report sprintf('- Mean spatial correlation: %.3f\n', ia.mean_spatial_correlation)];
        report = [report sprintf('- Mean temporal correlation: %.3f\n\n', ia.mean_temporal_correlation)];
    end
    
    report = [report sprintf('=====================================\n')];
end