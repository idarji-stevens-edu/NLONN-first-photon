function M = response_matrices(tomo)
% RESPONSE_MATRICES  Per-digit 10 x 10 response matrices from the tomography (Appendix B, Eq. B3).
%   Diagonal: the single-mode rates. Off-diagonal: the fringe of each mode pair over its phase steps,
%   R = A + B cos(phi) + C sin(phi), gives M_ab = s_ab^2 (B + iC) / (2 r).
    d = size(tomo.mode_mn, 1); n = size(tomo.rate, 2);
    M = complex(zeros(d, d, n));
    single = strcmp(tomo.mask_kind, 'single');
    for a = 1:d
        t = find(single & tomo.mask_mode_a == a);
        M(a, a, :) = reshape(tomo.rate(t, :), 1, 1, n);
    end
    ph = tomo.pair_phases_deg(:) * pi / 180;
    X = [ones(size(ph)) cos(tomo.phase_factor * ph) sin(tomo.phase_factor * ph)];
    for a = 1:d
        for b = a + 1:d
            t = find(~single & tomo.mask_mode_a == a & tomo.mask_mode_b == b);
            if isempty(t), continue, end
            [~, o] = sort(tomo.mask_phase_deg(t)); t = t(o);
            fit = X \ tomo.rate(t, :);                                  % A, B, C for every digit
            v = tomo.mask_scale(t(1))^2 / (2 * tomo.pair_scale) * (fit(2, :) + 1i * fit(3, :));
            M(a, b, :) = reshape(v, 1, 1, n);
            M(b, a, :) = reshape(conj(v), 1, 1, n);
        end
    end
end
