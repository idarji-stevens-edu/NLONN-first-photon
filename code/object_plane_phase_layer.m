%% Phase layer trained at the object plane, then a lens and a camera (Appendix A, Fig. 6)
% Alternates Adam steps on the phases (Eqs. A2, A3) with refitting the pixel-to-class map.
% Full scale takes about 25 min.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
mn = load(fullfile('..', 'data', 'mnist', 'MNIST.mat'));
test = load(fullfile('..', 'data', 'three_class_order3', 'test.mat'));
classes = test.classes;
ytr = double(mn.training.labels(:))'; yte = double(mn.test.labels(:))';
X = mn.training.images;
geo = signal_geometry(); G = geo.G;
rs = RandStream('mt19937ar', 'Seed', 1);
n_epochs = 30; batch = 256; lr = 0.05;

%% Training and validation images (90 / 10 % of the subset's training set)
itr = class_indices(ytr, classes, Inf);
itr = itr(randperm(rs, numel(itr)));
n_val = round(0.1 * numel(itr)); ival = itr(1:n_val); itr = itr(n_val + 1:end);
fprintf('%d training, %d validation images; %d x %d phases\n', numel(itr), n_val, G, G);

%% Check the gradient against finite differences
th = zeros(G);
d = pixel_class_map(X, itr(1:600), ytr, th, geo, 'lens', classes);
a = digit_fields(X(:, :, itr(1:8)), geo); y8 = ytr(itr(1:8));
th_t = 0.3 * randn(rs, G);
[J0, g0] = phase_layer_objective(a, y8, th_t, d, geo, 'lens');
lit = find(geo.amp > 0.3 * max(geo.amp(:)));        % pixels in the illuminated region
h = 1e-5; err = 0;
for t = 1:6
    p = lit(randi(rs, numel(lit))); dt = zeros(G); dt(p) = h;
    Jp = phase_layer_objective(a, y8, th_t + dt, d, geo, 'lens');
    Jm = phase_layer_objective(a, y8, th_t - dt, d, geo, 'lens');
    err = max(err, abs((Jp - Jm) / (2 * h) - g0(p)) / max(abs(g0(p)), 1e-12));
end
fprintf('Gradient vs finite differences: max relative error %.1e\n', err);

%% Train
d = pixel_class_map(X, itr, ytr, th, geo, 'lens', classes);   % theta = 0: the lens and camera
m1 = zeros(G); m2 = zeros(G); step = 0;
hist = zeros(n_epochs + 1, 3);
hist(1, :) = [0, mean(score_images(X, itr, ytr(itr), th, d, geo, 'lens')), mean(score_images(X, ival, ytr(ival), th, d, geo, 'lens'))];
best_val = -Inf;
for ep = 1:n_epochs
    order = itr(randperm(rs, numel(itr)));
    for s = 1:batch:numel(order)
        i = order(s:min(s + batch - 1, end));
        [~, g] = phase_layer_objective(digit_fields(X(:, :, i), geo), ytr(i), th, d, geo, 'lens');
        step = step + 1;
        m1 = 0.9 * m1 + 0.1 * g; m2 = 0.999 * m2 + 0.001 * g.^2;           % Adam, ascent
        th = th + lr * (m1 / (1 - 0.9^step)) ./ (sqrt(m2 / (1 - 0.999^step)) + 1e-12);
    end
    d = pixel_class_map(X, itr, ytr, th, geo, 'lens', classes);
    hist(ep + 1, :) = [ep, mean(score_images(X, itr, ytr(itr), th, d, geo, 'lens')), mean(score_images(X, ival, ytr(ival), th, d, geo, 'lens'))];
    if hist(ep + 1, 3) > best_val, best_val = hist(ep + 1, 3); best_th = th; best_d = d; best_ep = ep; end
    fprintf('Epoch %2d: training %.2f %%, validation %.2f %%\n', ep, 100 * hist(ep + 1, 2), 100 * hist(ep + 1, 3));
end

%% Test: the epoch best on validation
acc = score_images(mn.test.images, test.mnist_index, test.label, best_th, best_d, geo, 'lens');
ite = class_indices(yte, classes, Inf);
acc_all = mean(score_images(mn.test.images, ite, yte(ite), best_th, best_d, geo, 'lens'));
fprintf('Kept epoch %d: %.1f %% on the 450 test images, %.1f %% on all %d test images of the subset\n', ...
    best_ep, 100 * mean(acc), 100 * acc_all, numel(ite));
writetable(table(test.mnist_index(:), test.label(:), acc(:), 'VariableNames', {'mnist_index', 'label', 'object_plane_layer'}), ...
    fullfile('..', 'results', 'object_plane_layer_images.csv'));
writetable(array2table(hist, 'VariableNames', {'epoch', 'training', 'validation'}), fullfile('..', 'results', 'object_plane_layer_training.csv'));
