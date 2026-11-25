function [f, gr] = mask_gradient(w, Z, Y)
% MASK_GRADIENT  Accuracy behind mask w and its gradient in w (rule held at its argmax).
    n = size(Z, 2); p = w' * Z; ip = zeros(size(p)); ip(p > 0) = 1 ./ p(p > 0);
    S = Z * (Y .* ip)';
    [~, d] = max(S, [], 2);
    hit = Y(d, :); a = sum((w .* Z) .* hit, 1);
    f = double(sum(a .* ip)) / n;
    gr = double(sum(Z .* hit .* ip, 2) - Z * (a .* ip.^2)') / n;
end
