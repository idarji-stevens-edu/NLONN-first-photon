%% Acquisitions flagged by the reference kernel (Appendix C)
% An acquisition is flagged when the interpolated reference lies beyond +-15 % of the run's start.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
names = {'three_class_order1', 'three_class_order2', 'three_class_order3', 'ten_class_order2'};
tau = 100e-6;

for k = 1:numel(names)
    folder = fullfile('..', 'data', names{k});
    ref  = load(fullfile(folder, 'references.mat'));
    test = load(fullfile(folder, 'test.mat'));
    n_fail = nnz(~ref.ref_ok);
    n_train = nnz(ref.flagged_tomography) + nnz(ref.flagged_library);
    n_tr_all = numel(ref.flagged_tomography) + numel(ref.flagged_library);

    % accuracy with and without the test images a flagged acquisition touches
    [clicks, kernel_class] = load_clicks(folder, numel(test.classes));
    first = first_click_race(clicks, kernel_class, test.label, tau, test.t_int);
    one = double(first == test.label);
    hit = any(ref.flagged_test, 1);
    fprintf('%s: references failed %d | test flagged %d of %d acquisitions (%d images) | training %d of %d | single shot %.2f %% -> %.2f %% without them\n', ...
        names{k}, n_fail, nnz(ref.flagged_test), numel(ref.flagged_test), nnz(hit), n_train, n_tr_all, ...
        100 * mean(one), 100 * mean(one(~hit)));
end
