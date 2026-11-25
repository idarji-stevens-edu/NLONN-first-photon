%% Bound for detectors that record only which pixel the photon reaches, Eq. (A1) (Appendix A)
% Raw MNIST test set, photon probability proportional to brightness (Fischbacher & Sbaiz).
clear; clc;
cd(fileparts(mfilename('fullpath')));
mn = load(fullfile('..', 'data', 'mnist', 'MNIST.mat'), 'test');
X = reshape(double(mn.test.images), 784, []);
y = double(mn.test.labels(:))';

for classes = {0:9, [2 4 7]}
    c = classes{1};
    keep = ismember(y, c);
    W = X(:, keep) ./ sum(X(:, keep), 1);                 % each image's photon distribution
    P = zeros(784, numel(c));
    for q = 1:numel(c), P(:, q) = sum(W(:, y(keep) == c(q)), 2) / nnz(keep); end   % P(c, k)
    fprintf('Classes %s: bound sum_k max_c P(c, k) = %.2f %% (%d test images)\n', mat2str(c), 100 * sum(max(P, [], 2)), nnz(keep));
end
