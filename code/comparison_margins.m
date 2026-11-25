%% NLONN against linear optics on the same 450 test images (Sec. IV B, Fig. 4)
% Reads the per-image results of the other scripts; margins are paired per image, +- 1 standard error.
clear; clc;
cd(fileparts(mfilename('fullpath')));
res = fullfile('..', 'results');
nl = readtable(fullfile(res, 'three_class_order3_images.csv'));
lf = readtable(fullfile(res, 'lens_and_fixed_filter_images.csv'));
op = readtable(fullfile(res, 'object_plane_layer_images.csv'));
fp = readtable(fullfile(res, 'fourier_plane_layer_images.csv'));
mc = readtable(fullfile(res, 'masked_camera_images.csv'));
assert(isequal(nl.mnist_index, lf.mnist_index, op.mnist_index, fp.mnist_index, mc.mnist_index), 'different test images');

single = double(nl.single_shot_class == nl.label)';   % one recorded photon per image
trials = nl.trial_accuracy';
n = numel(single);
se = @(v) std(v) / sqrt(numel(v));

names = {'lens', 'fixed_pi_filter', 'object_plane_layer', 'fourier_plane_layer', 'purity_mask', 'trained_mask'};
lines = {lf.lens_camera', lf.fixed_filter', op.object_plane_layer', fp.fourier_plane_layer', mc.purity_mask', mc.trained_mask'};
fprintf('NLONN: single shot %.1f +- %.1f %%, repeated trials %.1f +- %.1f %% (%d images)\n', ...
    100 * mean(single), 100 * se(single), 100 * mean(trials), 100 * se(trials), n);
for k = 1:numel(names)
    c = lines{k};
    fprintf('%-20s %5.1f %% | NLONN - line: single shot %+5.1f +- %.1f pp, trials %+5.1f +- %.1f pp\n', names{k}, 100 * mean(c), ...
        100 * mean(single - c), 100 * se(single - c), 100 * mean(trials - c), 100 * se(trials - c));
end

%% Fig. 4 data, with the ideal NLONN (ten modes, all 3042 test images; run ideal_nlonn.m first)
ideal = readtable(fullfile(res, 'ideal_nlonn_orders.csv'));
line = [names(1:5), {'nlonn_trials', 'nlonn_single_shot', 'ideal_nlonn'}]';
acc = [100 * [cellfun(@mean, lines(1:5)), mean(trials), mean(single)], ideal.accuracy(ideal.max_order == 3)]';
err = [zeros(5, 1); 100 * se(trials); 100 * se(single); 0];
measured = [zeros(5, 1); 1; 1; 0];
writetable(table(line, acc, err, measured), fullfile(res, 'comparison.csv'));
