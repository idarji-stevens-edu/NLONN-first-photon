function M = normalise_responses(M, floor_level)
% NORMALISE_RESPONSES  Positive part, unit trace, plus a floor on every kernel's rate.
    d = size(M, 1);
    for i = 1:size(M, 3)
        Mi = (M(:, :, i) + M(:, :, i)') / 2;
        [V, ev] = eig(Mi, 'vector');
        Mi = V * diag(max(ev, 0)) * V';
        M(:, :, i) = Mi / max(real(trace(Mi)), realmin) + floor_level * eye(d);
    end
end
