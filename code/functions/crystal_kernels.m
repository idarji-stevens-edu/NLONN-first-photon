function [B, idx] = crystal_kernels(p)
% CRYSTAL_KERNELS  Kernel kappa_mn of each pump mode HG_mn on the signal modulator plane, Eq. (2).
%   The pump mode and the fibre mode are focused into the crystal; at each depth z their product is
%   propagated back at the signal wavelength and Fourier transformed to the modulator plane, and the
%   slices are summed with e^{i dk z}. Separable in x and y.
%   B: N^2 x K^2 (column b = m + K n + 1), idx(b, :) = [m n].
    K = p.basis_order; N = p.N;

    % mode indices and the peak of each uploaded mask (the modulator writes amplitude <= 1)
    idx = zeros(K^2, 2); peak = zeros(1, K^2); b = 0;
    env = exp(-(p.X.^2 + p.Y.^2) / p.pump_w0^2);
    for n = 0:K-1
        for m = 0:K-1
            b = b + 1; idx(b, :) = [m n];
            mask = hermite(m, sqrt(2) * p.X / p.pump_w0) .* hermite(n, sqrt(2) * p.Y / p.pump_w0) .* env;
            peak(b) = max(abs(mask(:)));
        end
    end

    % geometry: the realised pump field is the mask times the illumination
    w_real = 1 / sqrt(1 / p.pump_w0^2 + 1 / p.w_ill^2);
    w_Fp = p.lambda_p * p.lens_f / (pi * w_real);            % pump modes at the focus
    zR_p = pi * w_Fp^2 * p.n_p / p.lambda_p;                 % beams spread at lambda/n inside the crystal
    w_coll = p.lambda_f * p.coupler_f / (pi * p.fiber_mfd / 2);
    w_fc = p.lambda_f * p.lens_f2 / (pi * w_coll);           % fibre mode imaged into the crystal (~45 um)
    zR_f = pi * w_fc^2 * p.n_f / p.lambda_f;
    k_s = 2 * pi * p.n_s / p.lambda_s;

    % H_m(sqrt2 x / w0) exp(-x^2/w_real^2) expanded in H_j(sqrt2 x / w_real) exp(-x^2/w_real^2)
    gam = w_real / p.pump_w0; A = zeros(K);
    for m = 0:K-1
        for kk = 0:floor(m/2)
            j = m - 2*kk;
            A(j+1, m+1) = gam^(m-2*kk) * (gam^2 - 1)^kk * factorial(m) / (factorial(kk) * factorial(j));
        end
    end

    % grids
    dz = p.L / p.n_z; z = (-p.L/2 + dz/2) : dz : (p.L/2 - dz/2);
    du = 2 * p.u_halfwidth / p.u_points; u = (-p.u_points/2 : p.u_points/2 - 1)' * du;
    x = p.pitch * linspace(-N/2 + 1, N/2, N);
    W = du * exp(-1i * 2*pi * (x(:) * u') / (p.lambda_s * p.lens_f));     % focal plane -> modulator
    kx = 2*pi * ifftshift((-p.u_points/2 : p.u_points/2 - 1)') / (p.u_points * du);

    % z integral
    B = zeros(N*N, K^2);
    phase_j = (-1i).^(0:K-1);
    for iz = 1:p.n_z
        phi = hg_beam(0, u, z(iz), w_fc, zR_f, p.lambda_f / p.n_f);                 % fibre mode
        Hp = zeros(p.u_points, K);
        for j = 0:K-1, Hp(:, j+1) = phase_j(j+1) * hg_beam(j, u, z(iz), w_Fp, zR_p, p.lambda_p / p.n_p); end
        q = conj(phi) .* (Hp * A);
        if z(iz) ~= 0, q = ifft(exp(-1i * kx.^2 * z(iz) / (2 * k_s)) .* fft(q, [], 1), [], 1); end   % back to the focus
        kap = W * q;                                                         % N x K
        B = B + dz * exp(1i * p.delta_k * z(iz)) * reshape(reshape(kap(:, idx(:, 2)' + 1), N, 1, K^2) ...
            .* reshape(kap(:, idx(:, 1)' + 1), 1, N, K^2), N*N, K^2);
    end
    B = B ./ peak;
end

function E = hg_beam(j, u, z, w0, zR, lambda)
% One transverse dimension of a Hermite-Gaussian beam at distance z from its waist (Gouy phase included).
    w = w0 * sqrt(1 + (z/zR)^2);
    curv = 2*pi/lambda * z / (2 * (z^2 + zR^2));
    E = sqrt(w0/w) .* hermite(j, sqrt(2)*u/w) .* exp(-u.^2/w^2) .* exp(1i * curv * u.^2) .* exp(-1i * (j + 0.5) * atan2(z, zR));
end
