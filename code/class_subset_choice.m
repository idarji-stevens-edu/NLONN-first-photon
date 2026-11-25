%% Choice of the class subset (Sec. V B): cross-validation on a ten-class tomography
% A ten-class tomography (4 modes, 20 training digits per class) was ranked over all 582 subsets of
% 3-5 classes at acquisition time; the subsets below are that ranking's six best plus {0 2 4 7}.
% Here kernels are designed on the training folds' response matrices and scored on each held-out
% digit's own matrix (4 class-balanced folds). No rendering error enters.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
tomo = load(fullfile('..', 'data', 'subset_choice', 'tomography.mat'));
M = response_matrices(tomo); y = tomo.label; d = size(M, 1);
subsets = {[0 2 4 6], [2 4 6 7 9], [0 2 4 6 7], [0 2 4 6 9], [2 4 7], [0 2 6 7 9], [0 2 4 7]};
% post-selected Fourier-plane camera on each subset (model screen, MNIST test digits)
camera_ps = [37.1 30.9 31.5 31.4 46.5 31.9 37.8];
n_folds = 4;

acc = zeros(1, numel(subsets));
for u = 1:numel(subsets)
    cs = subsets{u}; m = find(ismember(y, cs)); ys = y(m); Ms = M(:, :, m); n = numel(m);
    fold = zeros(1, n);
    for c = cs, k = find(ys == c); fold(k) = mod(0:numel(k) - 1, n_folds) + 1; end
    p = zeros(1, n);
    for f = 1:n_folds
        tr = fold ~= f; te = fold == f;
        [C, w, dec] = design_kernels(Ms(:, :, tr), ys(tr), d, 4, 1, 0);
        Q = w .* kernel_rates(C, normalise_responses(Ms(:, :, te), 0));   % held-out digits' own rates
        P = Q ./ max(sum(Q, 1), realmin);
        p(te) = sum(P .* (dec(:) == ys(te)), 1);
    end
    acc(u) = mean(p);
    fprintf('{%s}: designed %.1f %% (chance %.1f %%), post-selected camera %.1f %%, margin %+.1f pp\n', ...
        num2str(cs), 100 * acc(u), 100 / numel(cs), camera_ps(u), 100 * acc(u) - camera_ps(u));
end
[~, best] = max(100 * acc - camera_ps);
fprintf('Largest margin over the camera: {%s}\n', num2str(subsets{best}));
