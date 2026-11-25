function geo = signal_geometry()
% SIGNAL_GEOMETRY  Simulation grid of the signal arm (Appendix A).
%   Digit 8.15 mm on 56 x 56 pixels (2 x 2 per MNIST pixel), 12 mm window (82 x 82 pixels of
%   145.5 um), Gaussian illumination of 2.83 mm 1/e^2 radius, Fourier transform zero-padded to 164 x 164.
    w = 2.83;                                   % illumination radius (mm)
    nsub = 2;                                   % simulation pixels per MNIST pixel
    px = 8.15 / 28 / nsub;                      % pixel (mm)
    G = 2 * round(12 / px / 2);                 % 82
    N = 2 * G;                                  % 164
    [xx, yy] = meshgrid(((1:G) - G/2 - 0.5) * px);
    [ku, kv] = meshgrid((1:N) - N/2 - 1);
    kr = hypot(ku, kv);                         % Fourier-plane radius in pixels, centred
    geo.G = G; geo.N = N; geo.nsub = nsub;
    geo.off = G/2 - 14 * nsub;                  % offset of the digit in the window
    geo.amp = exp(-(xx.^2 + yy.^2) / w^2);      % illumination amplitude
    geo.r2w = (xx.^2 + yy.^2) / w^2;            % (r / w)^2, for defocus
    geo.kr = kr;
    geo.kr_u = ifftshift(kr);                   % the same, in FFT order
    geo.spot_px = N * px / (pi * w);            % illumination's spot radius at the Fourier plane (pixels)
end
