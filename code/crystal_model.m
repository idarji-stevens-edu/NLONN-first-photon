%% Constants of the ideal NLONN model: crystal indices, fiber-mode waist, phase mismatch (Sec. III A, Appendix B)
clear; clc;
cd(fileparts(mfilename('fullpath')));
addpath('functions');
p = model_params();

%% Refractive indices at 40 C (Sellmeier fit) and the fiber mode's waist in the crystal
fprintf('Crystal index: %.4f pump, %.4f signal, %.4f sum frequency\n', ...
    crystal_index(p.lambda_p, 40), crystal_index(p.lambda_s, 40), crystal_index(p.lambda_f, 40));
w_coll = p.lambda_f * p.coupler_f / (pi * p.fiber_mfd / 2);
fprintf('Fiber mode in the crystal: waist %.1f um\n', 1e6 * p.lambda_f * p.lens_f2 / (pi * w_coll));

%% Phase mismatch at which HG00 converts a blank signal best
g = exp(-(p.X.^2 + p.Y.^2) / p.signal_w0^2); g = g(:) / norm(g(:));
p.basis_order = 1;                                   % HG00 only
dk = 0:5:150; rate = zeros(size(dk));
for q = 1:numel(dk)
    p.delta_k = dk(q); B = crystal_kernels(p); rate(q) = abs(g' * conj(B(:, 1)))^2;
end
[~, q] = max(rate); fine = dk(q) - 5 : 0.5 : dk(q) + 5; rf = zeros(size(fine));
for k = 1:numel(fine)
    p.delta_k = fine(k); B = crystal_kernels(p); rf(k) = abs(g' * conj(B(:, 1)))^2;
end
c = polyfit(fine - dk(q), rf / max(rf), 2);          % parabola through the peak
fprintf('HG00 rate peaks at delta_k = %.1f /m; at delta_k = 0 it is %.2f of the peak\n', dk(q) - c(2) / (2 * c(1)), rate(1) / max(rf));
