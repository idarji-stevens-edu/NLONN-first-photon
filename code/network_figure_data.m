%% Data for Fig. 1: one test image of the main result, its kernels, recorded clicks and rates
% The image is a typical correct decision that shows clicks under every kernel: true class 2 or 7,
% its first click names it, its kernel has the largest rate, every kernel clicks at least twice in
% the first 600 us, and its repeated-trial accuracy is nearest the median of its class.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
folder = fullfile('..', 'data', 'three_class_order3');
test = load(fullfile(folder, 'test.mat'));
lib = load(fullfile(folder, 'library.mat'));
mn = load(fullfile('..', 'data', 'mnist', 'MNIST.mat'), 'test');
[clicks, kernel_class] = load_clicks(folder, 3);
y = test.label; n = numel(y); t_int = test.t_int; tmax = 600e-6;
[first, trial_acc] = first_click_race(clicks, kernel_class, y, 100e-6, t_int);

%% Choose the image
n_all = zeros(3, n); n_early = zeros(3, n);
for q = 1:3
    for k = 1:n
        t = clicks{q, k}; n_all(q, k) = sum(t >= 0 & t < t_int); n_early(q, k) = sum(t >= 0 & t < tmax);
    end
end
[~, brightest] = max(n_all, [], 1);
best = inf; i = [];
for c = [2 7]
    cand = find(y == c & first == c & kernel_class(brightest) == c & all(n_early >= 2, 1));
    med = median(trial_acc(y == c), 'omitnan');
    [dist, k] = min(abs(trial_acc(cand) - med));
    if ~isempty(dist) && dist < best, best = dist; i = cand(k); end
end

%% Save what the figure draws
cc = lib.mask_coeffs(:, test.selected_mask);         % kernels' HG coefficients (modes x 3)
dec = kernel_class(:); w = ones(3, 1);               % equal dwells
modes = lib.mode_mn;
mnist_index = test.mnist_index(i);
img = double(mn.test.images(:, :, mnist_index));
t1 = sort(clicks{1, i})'; t2 = sort(clicks{2, i})'; t3 = sort(clicks{3, i})';
rates = [sum(t1 < t_int), sum(t2 < t_int), sum(t3 < t_int)] / t_int;   % counts/s
fprintf('Test image %d (MNIST test %d), class %d: first click names %d, trials %.3f, rates %s counts/s\n', ...
    i, mnist_index, y(i), first(i), trial_acc(i), mat2str(round(rates)));
save(fullfile('..', 'results', 'network_figure.mat'), 'cc', 'w', 'dec', 'img', 't1', 't2', 't3', 'modes', 'mnist_index', 'rates', '-v7');
