%% The ideal NLONN: kernels drawn from the HG modes of order <= k, ideal rendering, coherent conversion
% (Sec. IV B, Sec. V, Figs. 4, 5, 9b). One kernel per class, trained on 1000 digits per class.
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
mn = load(fullfile('..', 'data', 'mnist', 'MNIST.mat'));
test = load(fullfile('..', 'data', 'three_class_order3', 'test.mat'));
p = model_params(); N = p.N;

%% Mode kernels kappa_mn on the signal modulator plane, Eq. (2)
[B, idx] = crystal_kernels(p);
fprintf('%d HG modes (m, n <= %d)\n', size(idx, 1), max(idx(:)));

%% Signal states in a basis of the digit's pixel blocks (Eq. 1 on the model grid)
sz = round(784 * 10.4e-6 / p.pitch);                 % digit width in grid points
pre = floor((N - sz) / 2);
map = zeros(N); map(pre + (1:sz), pre + (1:sz)) = imresize(reshape(1:784, 28, 28), [sz sz], 'nearest');
map = map(:);                                        % MNIST pixel at each grid point (0 outside)
g = exp(-(p.X.^2 + p.Y.^2) / p.signal_w0^2); g = g(:) / norm(g(:));   % signal illumination

for task = 1:2
    if task == 1, classes = [2 4 7]; ks = 0:10; else, classes = 0:9; ks = 2; end
    [Xtr, ytr, itr] = mnist_subset(mn.training, classes, 1000);
    [Xte, yte, ite] = mnist_subset(mn.test, classes, Inf);

    % block basis: each lit MNIST pixel, plus the rest of the beam
    blk = find(any(Xte > 0, 2) | any(Xtr > 0, 2))'; nb = numel(blk);
    Bk = sparse(N^2, nb + 1);
    for q = 1:nb, s = map == blk(q); Bk(s, q) = g(s) / norm(g(s)); end
    s = ~ismember(map, blk); Bk(s, nb + 1) = g(s) / norm(g(s));
    w = abs(full(Bk' * g))'.^2;
    Atr = [sqrt(w(1:end-1))' .* exp(1i * pi * Xtr(blk, :)); sqrt(w(end)) * ones(1, size(Xtr, 2))]; Atr = Atr ./ vecnorm(Atr);
    Ate = [sqrt(w(1:end-1))' .* exp(1i * pi * Xte(blk, :)); sqrt(w(end)) * ones(1, size(Xte, 2))]; Ate = Ate ./ vecnorm(Ate);
    K = full(Bk' * conj(B)) / norm(B(:, 1));         % HG00 at full efficiency, others at their ratio

    acc = zeros(size(ks)); click = zeros(size(ks)); n_modes = zeros(size(ks));
    for t = 1:numel(ks)
        c = find(sum(idx, 2) <= ks(t)); n_modes(t) = numel(c);
        Gtr = K(:, c)' * Atr; Gte = K(:, c)' * Ate;   % projections g(x) onto the modes
        if numel(c) == 1                               % one mode: every kernel is HG00
            acc(t) = 1 / numel(classes); click(t) = mean(abs(Gte).^2); continue
        end
        V0 = complex(zeros(numel(c), numel(classes)));
        for j = 1:numel(classes)
            a = Gtr(:, ytr == classes(j)); [U, ~] = eigs(a * a', 1); V0(:, j) = U;   % start: each class's top mode
        end
        [V, history] = optimise_race_kernels(Gtr, ytr, V0, 1500);
        Pt = abs(V' * Gte).^2; click(t) = mean(sum(Pt, 1));
        pc = first_click_prob(Pt, yte, classes); acc(t) = mean(pc);
        fprintf('Classes %s, order <= %2d (%2d modes): %.1f %% on %d test images, %.3f %% of photons converted\n', ...
            mat2str(classes), ks(t), n_modes(t), 100 * acc(t), numel(yte), 100 * click(t));
        if task == 1 && ks(t) == 3
            [~, pos] = ismember(test.mnist_index, ite);
            fprintf('   on the 450 test images of the experiment: %.1f %%\n', 100 * mean(pc(pos)));
            writetable(table((0:numel(history) - 1)', history(:), 'VariableNames', {'step', 'training_accuracy'}), ...
                fullfile('..', 'results', 'ideal_nlonn_training.csv'));
        end
    end
    if task == 1
        writetable(table(ks(:), n_modes(:), 100 * acc(:), click(:), 'VariableNames', {'max_order', 'n_modes', 'accuracy', 'click_fraction'}), ...
            fullfile('..', 'results', 'ideal_nlonn_orders.csv'));
    end
end

function [X, y, idx] = mnist_subset(D, classes, n)
% First n images of each class, as 784 x images gray levels in [0, 1].
    lab = double(D.labels(:))'; idx = [];
    for c = classes, f = find(lab == c); idx = [idx f(1:min(n, numel(f)))]; end
    X = reshape(double(D.images(:, :, idx)), 784, []); if max(X(:)) > 1, X = X / 255; end
    y = lab(idx);
end
