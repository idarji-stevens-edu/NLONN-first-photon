function [acc_hard, acc_soft, dec] = design_accuracy(x, d, K, M, Y, beta)
% DESIGN_ACCURACY  Training accuracy of a design, Eq. (B1): hard, and with the soft decision of Eq. (B2).
%   Y: digits x 10 one-hot labels (classes 0-9); dec: class each kernel names
    [C, w] = unpack_kernels(x, d, K);
    Q = w .* kernel_rates(C, M);
    P = Q ./ max(sum(Q, 1), realmin);                % P(first click from kernel j | digit)
    L = (P * Y) ./ max(sum(Y, 1), 1);                % K x 10: class means of P
    [~, dec] = max(L, [], 2);
    acc_hard = mean(sum(P .* (((dec - 1) == (0:9)) * Y.'), 1));
    Z = beta * L ./ max(max(L, [], 2), realmin);
    Z = exp(Z - max(Z, [], 2)); Z = Z ./ sum(Z, 2);  % soft decision
    acc_soft = mean(sum(P .* (Z * Y.'), 1));
    dec = dec - 1;
end
