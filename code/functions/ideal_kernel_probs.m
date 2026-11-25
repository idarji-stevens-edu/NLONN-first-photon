function [P_test, P_train, y_train] = ideal_kernel_probs(mn, classes, mnist_index, max_order)
% IDEAL_KERNEL_PROBS  Kernel probabilities of the ideal NLONN (as ideal_nlonn.m, one mode set).
%   One kernel per class, trained on 1000 training digits per class within the HG modes of order
%   <= max_order. P_test: kernels x images for the test digits mnist_index; P_train: the same for the
%   training digits (labels y_train). Columns are normalized: the share of detected photons per kernel.
    p = model_params(); N = p.N;
    [B, idx] = crystal_kernels(p);
    sz = round(784 * 10.4e-6 / p.pitch); pre = floor((N - sz) / 2);
    map = zeros(N); map(pre + (1:sz), pre + (1:sz)) = imresize(reshape(1:784, 28, 28), [sz sz], 'nearest');
    map = map(:);
    g = exp(-(p.X.^2 + p.Y.^2) / p.signal_w0^2); g = g(:) / norm(g(:));
    [Xtr, y_train] = subset(mn.training, classes, 1000);
    [Xte, ~, ite] = subset(mn.test, classes, Inf);
    blk = find(any(Xte > 0, 2) | any(Xtr > 0, 2))'; nb = numel(blk);
    Bk = sparse(N^2, nb + 1);
    for q = 1:nb, s = map == blk(q); Bk(s, q) = g(s) / norm(g(s)); end
    s = ~ismember(map, blk); Bk(s, nb + 1) = g(s) / norm(g(s));
    w = abs(full(Bk' * g))'.^2;
    Atr = [sqrt(w(1:end-1))' .* exp(1i * pi * Xtr(blk, :)); sqrt(w(end)) * ones(1, size(Xtr, 2))]; Atr = Atr ./ vecnorm(Atr);
    Ate = [sqrt(w(1:end-1))' .* exp(1i * pi * Xte(blk, :)); sqrt(w(end)) * ones(1, size(Xte, 2))]; Ate = Ate ./ vecnorm(Ate);
    K = full(Bk' * conj(B)) / norm(B(:, 1));
    c = find(sum(idx, 2) <= max_order);
    Gtr = K(:, c)' * Atr; Gte = K(:, c)' * Ate;
    V0 = complex(zeros(numel(c), numel(classes)));
    for j = 1:numel(classes)
        a = Gtr(:, y_train == classes(j)); [U, ~] = eigs(a * a', 1); V0(:, j) = U;
    end
    V = optimise_race_kernels(Gtr, y_train, V0, 1500);
    [~, pos] = ismember(mnist_index, ite);
    P_test = abs(V' * Gte(:, pos)).^2; P_test = P_test ./ sum(P_test, 1);
    P_train = abs(V' * Gtr).^2; P_train = P_train ./ sum(P_train, 1);
end

function [X, y, idx] = subset(D, classes, n)
% First n images of each class, as 784 x images gray levels in [0, 1].
    lab = double(D.labels(:))'; idx = [];
    for c = classes, f = find(lab == c); idx = [idx f(1:min(n, numel(f)))]; end
    X = reshape(double(D.images(:, :, idx)), 784, []); if max(X(:)) > 1, X = X / 255; end
    y = lab(idx);
end
