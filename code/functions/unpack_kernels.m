function [C, w] = unpack_kernels(x, d, K)
% UNPACK_KERNELS  Optimiser vector [Re C; Im C; log w] -> unit-norm kernels C (d x K) and dwells w.
    C = reshape(x(1:d*K), d, K) + 1i * reshape(x(d*K+1:2*d*K), d, K);
    C = C ./ max(vecnorm(C), realmin);
    w = exp(x(2*d*K+1:end)); w = w(:);
end
