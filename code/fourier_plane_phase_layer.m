%% Phase layer trained at the Fourier plane of a 4f system, camera at the image plane (Appendix A, Fig. 6)
% A trained convolution. Starts from the best fixed filter (pi plate of one Fourier spot).
% Full scale takes about 45 min.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
mn = load(fullfile('..', 'data', 'mnist', 'MNIST.mat'));
test = load(fullfile('..', 'data', 'three_class_order3', 'test.mat'));
classes = test.classes;
ytr = double(mn.training.labels(:))'; yte = double(mn.test.labels(:))';
X = mn.training.images;
geo = signal_geometry(); N = geo.N;
rs = RandStream('mt19937ar', 'Seed', 1);
n_epochs = 30; batch = 256; lr = 0.05;

%% Training and validation images (90 / 10 % of the subset's training set)
itr = class_indices(ytr, classes, Inf);
itr = itr(randperm(rs, numel(itr)));
n_val = round(0.1 * numel(itr)); ival = itr(1:n_val); itr = itr(n_val + 1:end);
fprintf('%d training, %d validation images; %d x %d phases\n', numel(itr), n_val, N, N);

%% Start: pi plate on the zero order, radius one Fourier spot
plate = zeros(N); plate(geo.kr <= geo.spot_px) = pi;
phi = ifftshift(plate);

%% Check the gradient against finite differences
d = pixel_class_map(X, itr(1:600), ytr, phi, geo, 'fourier', classes);
a = digit_fields(X(:, :, itr(1:8)), geo); y8 = ytr(itr(1:8));
phi_t = phi + 0.3 * randn(rs, N);
[J0, g0] = phase_layer_objective(a, y8, phi_t, d, geo, 'fourier');
low = find(geo.kr_u <= 12);                         % Fourier pixels near the centre
h = 1e-5; err = 0;
for t = 1:6
    q = low(randi(rs, numel(low))); dt = zeros(N); dt(q) = h;
    Jp = phase_layer_objective(a, y8, phi_t + dt, d, geo, 'fourier');
    Jm = phase_layer_objective(a, y8, phi_t - dt, d, geo, 'fourier');
    err = max(err, abs((Jp - Jm) / (2 * h) - g0(q)) / max(abs(g0(q)), 1e-12));
end
fprintf('Gradient vs finite differences: max relative error %.1e\n', err);

%% Train
d = pixel_class_map(X, itr, ytr, phi, geo, 'fourier', classes);
m1 = zeros(N); m2 = zeros(N); step = 0;
hist = zeros(n_epochs + 1, 3);
best_val = mean(score_images(X, ival, ytr(ival), phi, d, geo, 'fourier'));
hist(1, :) = [0, mean(score_images(X, itr, ytr(itr), phi, d, geo, 'fourier')), best_val];
best_phi = phi; best_d = d; best_ep = 0;
for ep = 1:n_epochs
    order = itr(randperm(rs, numel(itr)));
    for s = 1:batch:numel(order)
        i = order(s:min(s + batch - 1, end));
        [~, g] = phase_layer_objective(digit_fields(X(:, :, i), geo), ytr(i), phi, d, geo, 'fourier');
        step = step + 1;
        m1 = 0.9 * m1 + 0.1 * g; m2 = 0.999 * m2 + 0.001 * g.^2;           % Adam, ascent
        phi = phi + lr * (m1 / (1 - 0.9^step)) ./ (sqrt(m2 / (1 - 0.999^step)) + 1e-12);
    end
    d = pixel_class_map(X, itr, ytr, phi, geo, 'fourier', classes);
    hist(ep + 1, :) = [ep, mean(score_images(X, itr, ytr(itr), phi, d, geo, 'fourier')), mean(score_images(X, ival, ytr(ival), phi, d, geo, 'fourier'))];
    if hist(ep + 1, 3) > best_val, best_val = hist(ep + 1, 3); best_phi = phi; best_d = d; best_ep = ep; end
    fprintf('Epoch %2d: training %.2f %%, validation %.2f %%\n', ep, 100 * hist(ep + 1, 2), 100 * hist(ep + 1, 3));
end

%% Test: the epoch best on validation
acc = score_images(mn.test.images, test.mnist_index, test.label, best_phi, best_d, geo, 'fourier');
ite = class_indices(yte, classes, Inf);
acc_all = mean(score_images(mn.test.images, ite, yte(ite), best_phi, best_d, geo, 'fourier'));
fprintf('Kept epoch %d: %.1f %% on the 450 test images, %.1f %% on all %d test images of the subset\n', ...
    best_ep, 100 * mean(acc), 100 * acc_all, numel(ite));
writetable(table(test.mnist_index(:), test.label(:), acc(:), 'VariableNames', {'mnist_index', 'label', 'fourier_plane_layer'}), ...
    fullfile('..', 'results', 'fourier_plane_layer_images.csv'));
writetable(array2table(hist, 'VariableNames', {'epoch', 'training', 'validation'}), fullfile('..', 'results', 'fourier_plane_layer_training.csv'));
