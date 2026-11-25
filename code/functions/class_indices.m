function idx = class_indices(labels, classes, n_max)
% CLASS_INDICES  Indices of the first n_max images of each class, class by class (row vector).
    idx = [];
    for c = classes
        k = find(labels(:) == c)';
        idx = [idx, k(1:min(n_max, numel(k)))];
    end
end
