%% All ten digits with kernels from the 6 HG modes of order <= 2 (Sec. V B)
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
folder = fullfile('..', 'data', 'ten_class_order2');
test = load(fullfile(folder, 'test.mat'));
tau = 100e-6;

[clicks, kernel_class] = load_clicks(folder, numel(test.classes));
[first, trial_acc] = first_click_race(clicks, kernel_class, test.label, tau, test.t_int);
one = double(first == test.label); n = numel(one);
fprintf('Ten classes, %d test images: single shot %.1f +- %.1f %%, repeated trials %.1f +- %.1f %%\n', ...
    n, 100 * mean(one), 100 * std(one) / sqrt(n), 100 * mean(trial_acc), 100 * std(trial_acc) / sqrt(n));
writetable(table(test.mnist_index(:), test.label(:), first(:), trial_acc(:), ...
    'VariableNames', {'mnist_index', 'label', 'single_shot_class', 'trial_accuracy'}), ...
    fullfile('..', 'results', 'ten_class_order2_images.csv'));
