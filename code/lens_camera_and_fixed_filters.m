%% Lens and camera, and fixed phase filters, on the 450 test images (Appendix A)
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
mn = load(fullfile('..', 'data', 'mnist', 'MNIST.mat'));
test = load(fullfile('..', 'data', 'three_class_order3', 'test.mat'));
classes = test.classes;
ytr = double(mn.training.labels(:))'; yte = double(mn.test.labels(:))';
idx = test.mnist_index; y = test.label;              % the 450 test images
geo = signal_geometry(); N = geo.N;

%% Lens + Fourier-plane camera, rule from all training images of the three classes
itr = class_indices(ytr, classes, Inf);
d = pixel_class_map(mn.training.images, itr, ytr, zeros(geo.G), geo, 'lens', classes);
acc_lens = score_images(mn.test.images, idx, y, zeros(geo.G), d, geo, 'lens');
fprintf('Lens + camera (full %d x %d grid): %.1f %%\n', N, N, 100 * mean(acc_lens));

%% Variants: a 64 x 64 crop with an outer bin and raw class sums; a rule from all test images
B = 64; c0 = N/2 + 1; r = c0 - B/2 + (0:B-1);
S_crop = zeros(B^2 + 1, numel(classes));
for s = 1:2000:numel(itr)
    i = itr(s:min(s + 1999, end));
    P = camera_distributions(digit_fields(mn.training.images(:, :, i), geo), zeros(geo.G), geo, 'lens');
    P = reshape(fftshift(fftshift(reshape(P, N, N, []), 1), 2), N, N, []);     % centred
    Pc = reshape(P(r, r, :), B^2, []); Pc = [Pc; 1 - sum(Pc, 1)];
    for q = 1:numel(classes), S_crop(:, q) = S_crop(:, q) + sum(Pc(:, ytr(i) == classes(q)), 2); end
end
[~, q] = max(S_crop, [], 2); d_crop = classes(q);
P = camera_distributions(digit_fields(mn.test.images(:, :, idx), geo), zeros(geo.G), geo, 'lens');
P = reshape(fftshift(fftshift(reshape(P, N, N, []), 1), 2), N, N, []);
Pc = reshape(P(r, r, :), B^2, []); Pc = [Pc; 1 - sum(Pc, 1)];
fprintf('Variant: 64 x 64 crop + outer bin, raw class sums: %.1f %%\n', 100 * mean(sum(Pc .* (d_crop(:) == y), 1)));
ite = class_indices(yte, classes, Inf);
d_test = pixel_class_map(mn.test.images, ite, yte, zeros(geo.G), geo, 'lens', classes);
fprintf('Variant: rule fitted on all %d test images of the subset: %.1f %%\n', numel(ite), ...
    100 * mean(score_images(mn.test.images, idx, y, zeros(geo.G), d_test, geo, 'lens')));

%% Seventeen fixed phase filters, chosen on 3000 training images per class
% defocus at the signal modulator, theta = beta pi (r/w)^2, camera at the Fourier plane;
% Zernike plate of phase phi on a disk of rho Fourier spots, camera at the image plane
i1 = class_indices(ytr, classes, 3000);
kind = {}; a = []; b = [];
for beta = [0.25 0.5 1 2 4], kind{end+1} = 'defocus'; a(end+1) = beta; b(end+1) = 0; end
for rho = [0.5 1 2 4], for phi = [pi/2 -pi/2 pi], kind{end+1} = 'zernike'; a(end+1) = phi; b(end+1) = rho; end, end
acc_train = zeros(1, 17); acc_test = zeros(17, numel(idx));
for s = 1:17
    if strcmp(kind{s}, 'defocus')
        mask = a(s) * pi * geo.r2w; cam = 'lens';
    else
        plate = zeros(N); plate(geo.kr <= b(s) * geo.spot_px) = a(s); mask = ifftshift(plate); cam = 'fourier';
    end
    [ds, Ss] = pixel_class_map(mn.training.images, i1, ytr, mask, geo, cam, classes);
    [~, q] = ismember(ds, classes);
    acc_train(s) = sum(Ss((1:N^2)' + N^2 * (q(:) - 1))) / numel(i1);       % training accuracy from the sums
    acc_test(s, :) = score_images(mn.test.images, idx, y, mask, ds, geo, cam);
    fprintf('   %-8s %5.2f, %3.1f: training %.2f %% | test %.2f %%\n', kind{s}, a(s) / pi^strcmp(kind{s}, 'zernike'), b(s), ...
        100 * acc_train(s), 100 * mean(acc_test(s, :)));
end
[~, best] = max(acc_train);
dfc = find(strcmp(kind, 'defocus')); [~, k] = max(acc_train(dfc)); best_defocus = dfc(k);
fprintf('Best fixed filter on training: %s phi = %.2f pi, rho = %.1f spots: %.1f %%\n', kind{best}, a(best) / pi, b(best), 100 * mean(acc_test(best, :)));
fprintf('Best defocus on training: beta = %.2f: %.1f %%\n', a(best_defocus), 100 * mean(acc_test(best_defocus, :)));

%% Save per-image accuracies
writetable(table(idx(:), y(:), acc_lens(:), acc_test(best, :)', 'VariableNames', {'mnist_index', 'label', 'lens_camera', 'fixed_filter'}), ...
    fullfile('..', 'results', 'lens_and_fixed_filter_images.csv'));
