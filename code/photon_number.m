%% Accuracy against the number of detected photons per decision (Appendix C, Fig. 10; ten-photon confusion matrix, Fig. 3b)
% Bayesian decision on k clicks: from equal priors, each click multiplies the probability of every
% class by the mean share of that class's training digits at the kernel that clicked; time slots
% without a click are ignored; after k clicks the most probable class is chosen. At k = 1 this is the
% first-click decision of Sec. IV A. Takes about 2 min.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
folder = fullfile('..', 'data', 'three_class_order3');
mn = load(fullfile('..', 'data', 'mnist', 'MNIST.mat'));
lib = load(fullfile(folder, 'library.mat'));
test = load(fullfile(folder, 'test.mat'));
classes = test.classes; label = test.label; n = numel(label);
ks = [1 2 3 4 5 7 10 15 20 30 50 70 100];
tau = 100e-6;                                        % length of one repeated trial (s), as Sec. IV A
rs = RandStream('mt19937ar', 'Seed', 7);
n_rep = 2000;                                        % simulated decisions per image and k

%% Measured NLONN: class shares from the 150 training digits, photons from the recorded trials
P_lib = max(lib.rate_recorded(test.selected_mask, :), 0); P_lib = P_lib ./ sum(P_lib, 1);
logS = log(class_shares(P_lib, lib.label, classes));
[clicks, kernel_class] = load_clicks(folder, numel(classes));
assert(isequal(kernel_class, classes), 'kernels and classes in different order');
acc = nan(n, numel(ks));
named10 = nan(n, numel(classes));                    % share of each class named per image at k = 10 (Fig. 3b)
for i = 1:n
    o = trial_kernels(clicks(:, i), tau, test.t_int);      % kernel of each trial's first click
    for q = 1:numel(ks)
        k = ks(q); nb = floor(numel(o) / k); if nb < 1, continue, end
        blocks = reshape(o(1:nb * k), k, nb);
        L = zeros(nb, numel(classes));
        for c = 1:numel(classes), L(:, c) = sum(reshape(logS(blocks, c), k, nb), 1)'; end
        [~, d] = max(L, [], 2); acc(i, q) = mean(classes(d) == label(i));
        if k == 10, for c = 1:numel(classes), named10(i, c) = mean(d == c); end, end
    end
end
nlonn = mean(acc, 1, 'omitnan'); nlonn_se = std(acc, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(acc), 1));
conf10 = zeros(numel(classes), numel(classes)); conf10_se = conf10;   % rows true class, columns class named
for c = 1:numel(classes)
    v = named10(label == classes(c), :);
    conf10(c, :) = mean(v, 1, 'omitnan'); conf10_se(c, :) = std(v, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(v), 1));
end
disp('Confusion at k = 10 (rows: true class, columns: named class):'); disp(round(conf10, 3));
fprintf('Largest standard error of an entry: %.3f\n', max(conf10_se(:)));
S = exp(logS); j4 = find(classes == 4);              % selectivity of the kernel naming 4 (Sec. IV A)
fprintf('A click of the kernel naming 4 is %s times more likely from a 4 than from a %s\n', ...
    mat2str(round(S(j4, j4) ./ S(j4, classes ~= 4), 1)), mat2str(classes(classes ~= 4)));

%% Ideal NLONN (ten modes of order <= 3): photons drawn from its kernel probabilities
[P_test, P_train, y_train] = ideal_kernel_probs(mn, classes, test.mnist_index, 3);
logS_id = log(class_shares(P_train, y_train, classes));
acc_id = zeros(n, numel(ks));
for i = 1:n
    cdf = cumsum(P_test(:, i)); cdf = cdf / cdf(end);
    o = discretize(rand(rs, max(ks), n_rep), [0; cdf]);
    for q = 1:numel(ks)
        L = zeros(n_rep, numel(classes));
        for c = 1:numel(classes), L(:, c) = sum(reshape(logS_id(o(1:ks(q), :), c), ks(q), n_rep), 1)'; end
        [~, d] = max(L, [], 2); acc_id(i, q) = mean(classes(d) == label(i));
    end
end
ideal = mean(acc_id, 1); ideal_se = std(acc_id, 0, 1) / sqrt(n);

fprintf('k:            %s\n', sprintf('%6d', ks));
fprintf('NLONN:        %s  (n = %d images)\n', sprintf('%6.1f', 100 * nlonn), min(sum(~isnan(acc), 1)));
fprintf('ideal NLONN:  %s\n', sprintf('%6.1f', 100 * ideal));
fprintf('NLONN: %.1f +- %.1f %% at k = 1, %.1f +- %.1f %% at 10, %.1f +- %.1f %% at 100\n', ...
    100 * nlonn(1), 100 * nlonn_se(1), 100 * nlonn(ks == 10), 100 * nlonn_se(ks == 10), 100 * nlonn(end), 100 * nlonn_se(end));

%% Ceiling of three outcomes: the exact kernel probabilities, classified by their 15 nearest training digits
[~, row] = ismember(test.selected_mask, test.test_mask);
P_meas = max(test.rate_recorded(row, :), 0); P_meas = P_meas ./ sum(P_meas, 1);
knn = @(Ptr, ytr, Pte) predict(fitcknn(Ptr', ytr(:), 'NumNeighbors', 15), Pte');
fprintf('Exact kernel probabilities, 15 nearest training digits: measured %.1f %%, ideal %.1f %%\n', ...
    100 * mean(knn(P_lib, lib.label, P_meas) == label(:)), 100 * mean(knn(P_train, y_train, P_test) == label(:)));

%% Save
writetable(table(ks(:), 100 * nlonn(:), 100 * nlonn_se(:), 100 * ideal(:), 100 * ideal_se(:), ...
    'VariableNames', {'photons', 'nlonn', 'nlonn_err', 'ideal_nlonn', 'ideal_nlonn_err'}), ...
    fullfile('..', 'results', 'photon_number.csv'));
names = compose('%d', classes(:));
writetable(array2table(conf10, 'RowNames', strcat('true_', names), 'VariableNames', strcat('named_', names)), ...
    fullfile('..', 'results', 'confusion_photons10.csv'), 'WriteRowNames', true);

function S = class_shares(P, label, classes)
% Mean share of each class's digits at each kernel (kernels x classes).
    S = zeros(size(P, 1), numel(classes));
    for c = 1:numel(classes), S(:, c) = mean(P(:, label == classes(c)), 2); end
end

function o = trial_kernels(clicks, tau, t_int)
% Kernel whose click comes first in each trial of length tau (trials without a click are skipped).
    n_seg = floor(t_int / tau); t_seg = inf(numel(clicks), n_seg);
    for j = 1:numel(clicks)
        t = sort(clicks{j}); t = t(t >= 0 & t < t_int); if isempty(t), continue, end
        seg = floor(t / tau) + 1; keep = seg <= n_seg; t = t(keep); seg = seg(keep);
        [s, k] = unique(seg, 'first'); t_seg(j, s) = t(k) - (s - 1) * tau;
    end
    [tmin, o] = min(t_seg, [], 1); o = o(isfinite(tmin));
end
