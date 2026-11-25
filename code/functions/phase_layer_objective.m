function [J, g] = phase_layer_objective(a, y, mask, d, geo, kind)
% PHASE_LAYER_OBJECTIVE  J = mean probability that the photon lands on a pixel of its own class,
%   Eq. (A2), and its gradient in the phase mask, Eq. (A3).
%   kind 'lens': mask theta at the object plane; 'fourier': mask phi at the Fourier plane.
    N = geo.N; G = geo.G; n = size(a, 3);
    W = reshape(d(:) == y(:)', N, N, n);              % pixels of the image's own class
    if strcmp(kind, 'lens')
        f = a .* exp(1i * mask);
        F = fft2(f, N, N);
        S = sum(abs(F).^2, [1 2]);                    % does not depend on theta (Parseval)
        J = mean(sum(W .* abs(F).^2, [1 2]) ./ S);
        if nargout > 1
            gF = N^2 * ifft2(W .* F); gF = gF(1:G, 1:G, :);     % back to the phase layer
            g = sum(-2 * imag(f .* conj(gF)) ./ S, 3) / n;
        end
    else
        F = fft2(a, N, N); HF = exp(1i * mask) .* F;
        u = ifft2(HF); I = abs(u).^2;
        T = sum(I, [1 2]);
        Ji = sum(W .* I, [1 2]) ./ T; J = mean(Ji);
        if nargout > 1
            B = fft2((W - Ji) .* u) / N^2;                       % back to the Fourier plane
            g = sum(-2 * imag(conj(B) .* HF) ./ T, 3) / n;
        end
    end
end
