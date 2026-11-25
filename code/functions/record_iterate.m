function out = record_iterate(x, state)
% RECORD_ITERATE  Keep the optimiser's iterates (used as the fminunc OutputFcn).
%   record_iterate([], 'reset') clears them; X = record_iterate([], 'get') returns them as columns.
    persistent X
    out = false;
    if strcmp(state, 'reset'), X = []; return, end
    if strcmp(state, 'get'), out = X; return, end
    if strcmp(state, 'iter'), X = [X, x(:)]; end
end
