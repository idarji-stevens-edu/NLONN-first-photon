function acc = score_images(X, idx, y, mask, d, geo, kind)
% SCORE_IMAGES  Probability that each image's photon lands on a pixel of its own class (map d).
%   y: labels of the images X(:,:,idx)
    acc = zeros(1, numel(idx));
    for s = 1:1000:numel(idx)
        q = s:min(s + 999, numel(idx));
        P = camera_distributions(digit_fields(X(:, :, idx(q)), geo), mask, geo, kind);
        acc(q) = sum(P .* (d(:) == y(q)), 1);
    end
end
