%% Cameras that lose photons: behind a purity-ranked mask and a trained mask (Appendix A)
% Fourier-plane camera on the full 164 x 164 grid; rules from 3000 training images per class.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
mn = load(fullfile('..', 'data', 'mnist', 'MNIST.mat'));
test = load(fullfile('..', 'data', 'three_class_order3', 'test.mat'));
classes = test.classes;
ytr = double(mn.training.labels(:))'; yte = double(mn.test.labels(:))';
idx = test.mnist_index; y = test.label;
geo = signal_geometry(); N = geo.N;
targets = logspace(-3, 0, 31);                       % mean acceptances tried

%% Photon distributions (centred pixel order)
itr = class_indices(ytr, classes, 3000); y_tr = ytr(itr);
Ztr = zeros(N^2, numel(itr), 'single');
for s = 1:2000:numel(itr)
    q = s:min(s + 1999, numel(itr));
    P = camera_distributions(digit_fields(mn.training.images(:, :, itr(q)), geo), zeros(geo.G), geo, 'lens');
    Ztr(:, q) = single(reshape(fftshift(fftshift(reshape(P, N, N, []), 1), 2), N^2, []));
end
P = camera_distributions(digit_fields(mn.test.images(:, :, idx), geo), zeros(geo.G), geo, 'lens');
Zte = reshape(fftshift(fftshift(reshape(P, N, N, []), 1), 2), N^2, []);
ite = class_indices(yte, classes, Inf); iout = ite(~ismember(ite, idx));      % other test images of the subset
P = camera_distributions(digit_fields(mn.test.images(:, :, iout), geo), zeros(geo.G), geo, 'lens');
Zout = reshape(fftshift(fftshift(reshape(P, N, N, []), 1), 2), N^2, []);

%% Purity-ranked mask; acceptance chosen on the other test images of the subset
[A, n_pix, purity, order] = masked_camera_purity(Ztr, y_tr, Zte, y, targets);
A_out = masked_camera_purity(Ztr, y_tr, Zout, yte(iout), targets);
[~, k] = max(mean(A_out, 2));
keep = order(1:n_pix(k));
pass = sum(Zte(keep, :), 1);
[~, brightest] = max(sum(double(Ztr), 2));
fprintf('Purity-ranked mask: %d pixels of purity %.2f-%.2f, passing %.2f %% of the photons: %.1f %%\n', ...
    n_pix(k), min(purity(keep)), max(purity(keep)), 100 * mean(pass), 100 * mean(A(k, :)));
fprintf('Brightest pixel (centre of the focal spot): purity %.2f; a pixel without class information: %.2f\n', ...
    purity(brightest), 1 / numel(classes));
acc_purity = A(k, :);

%% Trained transmission mask
[w, d] = masked_camera_trained(double(Ztr), y_tr);
d(isnan(d)) = -1;
pass = sum(w .* Zte, 1);
acc_trained = sum(w .* Zte .* (d(:) == y), 1) ./ max(pass, realmin); acc_trained(pass <= 0) = 0;
fprintf('Trained mask: %d pixels transmitting, passing %.3f %% of the photons: %.1f %%\n', nnz(w > 0), 100 * mean(pass), 100 * mean(acc_trained));

%% Save per-image accuracies
writetable(table(idx(:), y(:), acc_purity(:), acc_trained(:), 'VariableNames', {'mnist_index', 'label', 'purity_mask', 'trained_mask'}), ...
    fullfile('..', 'results', 'masked_camera_images.csv'));
