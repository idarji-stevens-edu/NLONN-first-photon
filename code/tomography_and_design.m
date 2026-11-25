%% Tomography, coherence and kernel design of the main result (Appendix B; Figs. 7, 8, 9a)
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
folder = fullfile('..', 'data', 'three_class_order3');
tomo = load(fullfile(folder, 'tomography.mat'));
res = fullfile('..', 'results');

%% Response matrices of the training digits
M = response_matrices(tomo);                         % d x d x digits, Eq. (B3)
d = size(M, 1); n = size(M, 3); label = tomo.label;
Mn = normalise_responses(M, 0);                      % positive part, unit trace
fprintf('Tomography: %d masks on %d training digits, %d modes\n', size(tomo.rate, 1), n, d);

%% Coherence: eigenvalues of each M_x / Tr M_x (Fig. 8)
ev = zeros(n, d);
for i = 1:n, ev(i, :) = sort(real(eig((Mn(:, :, i) + Mn(:, :, i)') / 2)), 'descend')'; end
fprintf('Largest eigenvalue share: median %.3f (IQR %.3f-%.3f)\n', median(ev(:, 1)), prctile(ev(:, 1), 25), prctile(ev(:, 1), 75));
writematrix([label(:), ev], fullfile(res, 'tomography_eigenvalues.csv'));

%% Class means of M_x / Tr M_x (Fig. 7)
T = [];
[a, b] = ndgrid(1:d, 1:d);
for c = tomo.classes
    Mc = mean(Mn(:, :, label == c), 3);
    T = [T; repmat(c, d^2, 1), a(:), b(:), real(Mc(:)), imag(Mc(:))];
end
writematrix(T, fullfile(res, 'tomography_class_means.csv'));
writematrix([(1:d)', tomo.mode_mn], fullfile(res, 'tomography_modes.csv'));

%% Kernel design at two floors (Fig. 9a)
H = [];
for floor_level = [0 0.01]
    [C, w, dec, acc, history] = design_kernels(M, label, 10, 4, 37, floor_level);
    fprintf('Design with floor %.2f: predicted training accuracy %.1f %% (%d iterations)\n', floor_level, 100 * acc, size(history, 1));
    H = [H; repmat(floor_level, size(history, 1), 1), (1:size(history, 1))', history];
end
writematrix(H, fullfile(res, 'design_history.csv'));

%% Measured against predicted rates of the designed library masks (Appendix B, Selection)
% class-mean rate relative to HG00: measured on the library digits, predicted from the tomography
lib = load(fullfile(folder, 'library.mat'));
test = load(fullfile(folder, 'test.mat'));
Mp = M;
for i = 1:n                                          % positive part of each response matrix
    Mi = (M(:, :, i) + M(:, :, i)') / 2; [V, ev] = eig(Mi, 'vector');
    Mp(:, :, i) = V * diag(max(ev, 0)) * V';
end
pred = kernel_rates(lib.mask_coeffs, Mp);            % library masks x training digits
hg00 = find(strcmp(lib.mask_kind, 'single') & lib.mask_mode == 1);
ratio = zeros(size(lib.rate, 1), numel(lib.classes));
for q = 1:numel(lib.classes)
    p = mean(pred(:, label == lib.classes(q)), 2);   p = p / p(hg00);
    m = mean(lib.rate(:, lib.label == lib.classes(q)), 2); m = m / m(hg00);
    ratio(:, q) = m ./ p;
end
designed = strcmp(lib.mask_kind, 'designed');
r = ratio(designed, :);
fprintf('Designed masks, measured / predicted class-mean rate (rel. HG00): %.2f-%.2f, median %.2f\n', min(r(:)), max(r(:)), median(r(:)));
for k = 1:numel(test.selected_mask)
    fprintf('Kernel naming %d (mask %d): %s on classes %s\n', test.selected_class(k), test.selected_mask(k), ...
        mat2str(round(ratio(test.selected_mask(k), :), 2)), mat2str(lib.classes));
end
