function Q = kernel_rates(C, M)
% KERNEL_RATES  Predicted rate of each kernel on each digit, q_j(i) = c_j.' M_i conj(c_j).
    K = size(C, 2); n = size(M, 3); Q = zeros(K, n);
    for j = 1:K
        Q(j, :) = real(reshape(pagemtimes(pagemtimes(C(:, j).', M), conj(C(:, j))), 1, n));
    end
    Q = max(Q, 0);
end
