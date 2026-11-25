%% Main result: digits 2, 4, 7 with kernels from the 10 HG modes of order <= 3 (Sec. IV A)
% Selection on the library, then the test decided from recorded clicks.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
folder = fullfile('..', 'data', 'three_class_order3');
lib  = load(fullfile(folder, 'library.mat'));
test = load(fullfile(folder, 'test.mat'));
classes = test.classes;
tau = 100e-6;                                       % length of one repeated trial (s)

%% Kernel selection on the library digits (Appendix B, Selection)
% chosen among the masks the test measured, and checked against all library masks
[idx, acc_train, acc_cv, se_cv] = select_kernels(lib.rate(test.test_mask, :), lib.label, classes);
assert(isequal(test.test_mask(idx), test.selected_mask), 'selection differs from the tested kernels');
idx_all = select_kernels(lib.rate, lib.label, classes);
n_lib = numel(lib.label);
p_tr = first_click_prob(lib.rate(test.selected_mask, :), lib.label, classes);
fprintf('Library: %d masks on %d digits; the test measured %d\n', size(lib.rate, 1), n_lib, numel(test.test_mask));
fprintf('Trained kernels: masks %s naming %s; same choice among all masks: %d\n', ...
    mat2str(test.selected_mask), mat2str(classes), isequal(idx_all, test.selected_mask));
fprintf('Training accuracy %.1f +- %.1f %%, nested 5-fold CV %.1f +- %.1f %%\n', ...
    100 * acc_train, 100 * std(p_tr) / sqrt(n_lib), 100 * acc_cv, 100 * se_cv);

%% Count rates in the test
r = test.rate_recorded;                              % detected photons per ms
fprintf('Test rates: %.2f to %.0f counts/ms (%.0e to %.0e photons per pulse at 100 MHz)\n', ...
    min(r(:)), max(r(:)), min(r(:)) / 1e5, max(r(:)) / 1e5);

%% Accuracy from recorded clicks
[clicks, kernel_class] = load_clicks(folder, numel(classes));
label = test.label; n = numel(label);
[first, trial_acc, n_trials] = first_click_race(clicks, kernel_class, label, tau, test.t_int);
single = double(first == label);                     % one recorded photon per image
fprintf('Single shot: %.1f +- %.1f %% (%d of %d images decided)\n', 100 * mean(single), 100 * std(single) / sqrt(n), nnz(~isnan(first)), n);
fprintf('Repeated trials (%.0f us): %.1f +- %.1f %%, median %d trials per image\n', 1e6 * tau, ...
    100 * mean(trial_acc), 100 * std(trial_acc) / sqrt(n), median(n_trials));

%% Confusion matrix from the repeated trials (Fig. 3)
conf = zeros(numel(classes)); conf_se = conf;       % standard error over the images of each class
for q = 1:numel(classes)
    [~, named] = first_click_race(clicks, kernel_class, classes(q) * ones(1, n), tau, test.t_int);   % share of trials naming class q
    for c = 1:numel(classes)
        v = named(label == classes(c)); conf(c, q) = mean(v); conf_se(c, q) = std(v) / sqrt(numel(v));
    end
end
disp('Confusion (rows: true class, columns: named class):'); disp(round(conf, 3));
fprintf('Largest standard error of an entry: %.3f\n', max(conf_se(:)));

%% Mean rate of each kernel on each class (counts/ms)
[~, row] = ismember(test.selected_mask, test.test_mask);
rk = test.rate_recorded(row, :);
for c = 1:numel(classes)
    fprintf('Mean rates of the kernels naming %s on %ds: %s counts/ms\n', mat2str(classes), classes(c), ...
        mat2str(round(mean(rk(:, label == classes(c)), 2)', 1)));
end

%% Save per-image results and the confusion matrix
writetable(table(test.mnist_index(:), label(:), first(:), trial_acc(:), n_trials(:), ...
    'VariableNames', {'mnist_index', 'label', 'single_shot_class', 'trial_accuracy', 'n_trials'}), ...
    fullfile('..', 'results', 'three_class_order3_images.csv'));
writetable(array2table(conf, 'VariableNames', compose('named_%d', classes), 'RowNames', compose('true_%d', classes)), ...
    fullfile('..', 'results', 'three_class_order3_confusion.csv'), 'WriteRowNames', true);
