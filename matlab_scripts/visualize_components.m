function visualize_components(results, varargin)
% VISUALIZE_COMPONENTS Visualize NNMF spatial and temporal components
%
% Usage:
%   visualize_components(results)
%   visualize_components(results, 'n_components', 6)

% Parse inputs
p = inputParser;
addRequired(p, 'results', @isstruct);
addParameter(p, 'n_components', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
parse(p, results, varargin{:});

if ~isfield(results, 'final_model')
    error('Results must contain final_model field');
end

W = results.final_model.W;
H = results.final_model.H;
n_comp_total = size(W, 2);
n_comp_plot = p.Results.n_components;
if isempty(n_comp_plot)
    n_comp_plot = min(6, n_comp_total);
end

fprintf('Visualizing %d components (out of %d total)\n', n_comp_plot, n_comp_total);

% Create figure
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Spatial Components Heatmap
subplot(2, 3, 1);
imagesc(W(:, 1:n_comp_plot)');
colorbar;
colormap('RdBu_r');
title('Spatial Components (W)');
xlabel('Electrodes');
ylabel('Components');
set(gca, 'YTick', 1:n_comp_plot, 'YTickLabel', 1:n_comp_plot);

% Plot 2: Temporal Components
subplot(2, 3, 2);
colors = lines(n_comp_plot);
for i = 1:n_comp_plot
    plot(H(i, :), 'Color', colors(i, :), 'LineWidth', 1.5);
    hold on;
end
xlabel('Time');
ylabel('Component Activation');
title('Temporal Components (H)');
legend(arrayfun(@(x) sprintf('C%d', x), 1:n_comp_plot, 'UniformOutput', false), ...
       'Location', 'best');
grid on;

% Plot 3: Component Activations Heatmap
subplot(2, 3, 3);
imagesc(H(1:n_comp_plot, :));
colorbar;
colormap('viridis');
title('Component Activations');
xlabel('Time');
ylabel('Components');
set(gca, 'YTick', 1:n_comp_plot, 'YTickLabel', 1:n_comp_plot);

% Plot 4: Cross-validation Results
if isfield(results, 'cross_validation')
    subplot(2, 3, 4);
    cv = results.cross_validation;
    if isfield(cv, 'mean_recon_errors') && isfield(cv, 'component_range')
        plot(cv.component_range, mean(cv.mean_recon_errors, 2), 'o-', 'LineWidth', 2);
        hold on;
        xline(cv.optimal_components, 'r--', 'LineWidth', 2);
        xlabel('Number of Components');
        ylabel('Reconstruction Error');
        title('CV: Reconstruction Error');
        grid on;
    end
end

% Plot 5: Reliability
if isfield(results, 'cross_validation') && isfield(results.cross_validation, 'mean_reliability')
    subplot(2, 3, 5);
    cv = results.cross_validation;
    plot(cv.component_range, mean(cv.mean_reliability, 2), 'o-', 'Color', 'green', 'LineWidth', 2);
    hold on;
    xline(cv.optimal_components, 'r--', 'LineWidth', 2);
    xlabel('Number of Components');
    ylabel('Reliability');
    title('CV: Reliability');
    grid on;
end

% Plot 6: Component variance
subplot(2, 3, 6);
component_var = var(H(1:n_comp_plot, :), [], 2);
bar(1:n_comp_plot, component_var);
xlabel('Component');
ylabel('Temporal Variance');
title('Component Temporal Variance');

story_name = 'Unknown Story';
if isfield(results, 'metadata') && isfield(results.metadata, 'story_name')
    story_name = results.metadata.story_name;
end

sgtitle(sprintf('NNMF Components - %s', story_name));

end