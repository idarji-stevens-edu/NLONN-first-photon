%% Accuracy against the highest pump-mode order, digits 2, 4, 7 (Sec. V A, Fig. 5)
% Each experiment scored as the main result: one kernel per class, recorded clicks.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
names = {'three_class_order1', 'three_class_order2', 'three_class_order3'};
max_order = [1 2 3];
n_modes = [3 6 10];
tau = 100e-6;

single_acc = zeros(1, 3); single_se = zeros(1, 3); trial_acc = zeros(1, 3); trial_se = zeros(1, 3);
for k = 1:3
    folder = fullfile('..', 'data', names{k});
    test = load(fullfile(folder, 'test.mat'));
    [clicks, kernel_class] = load_clicks(folder, numel(test.classes));
    [first, acc_i] = first_click_race(clicks, kernel_class, test.label, tau, test.t_int);
    one = double(first == test.label); n = numel(one);
    single_acc(k) = 100 * mean(one);   single_se(k) = 100 * std(one) / sqrt(n);
    trial_acc(k)  = 100 * mean(acc_i); trial_se(k)  = 100 * std(acc_i) / sqrt(n);
    fprintf('Order <= %d (%d modes): single shot %.1f +- %.1f %%, repeated trials %.1f +- %.1f %% (%d images)\n', ...
        max_order(k), n_modes(k), single_acc(k), single_se(k), trial_acc(k), trial_se(k), n);
end
writetable(table(max_order(:), n_modes(:), single_acc(:), single_se(:), trial_acc(:), trial_se(:), ...
    'VariableNames', {'max_order', 'n_modes', 'single_shot', 'single_shot_se', 'trials', 'trials_se'}), ...
    fullfile('..', 'results', 'accuracy_vs_mode_order.csv'));
