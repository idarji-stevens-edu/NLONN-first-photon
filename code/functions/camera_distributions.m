function P = camera_distributions(a, mask, geo, kind)
% CAMERA_DISTRIBUTIONS  Where the photon lands on the camera (N^2 x n, columns sum to 1).
%   kind 'lens'    : phase mask at the object plane, then a lens, camera at the Fourier plane
%   kind 'fourier' : 4f system, phase mask at the Fourier plane, camera at the image plane
%   mask: phase (rad), G x G for 'lens', N x N in FFT order for 'fourier'. Pixels are in FFT order.
    N = geo.N;
    if strcmp(kind, 'lens')
        I = abs(fft2(a .* exp(1i * mask), N, N)).^2;
        P = reshape(I, N^2, []);
        P = P ./ sum(P, 1);
    else
        I = abs(ifft2(exp(1i * mask) .* fft2(a, N, N))).^2;
        P = reshape(I ./ sum(I, [1 2]), N^2, []);
    end
end
