function [copyed_T, copyed_F] = copySmallSample(train_T,train_F)
    % copy small train samples
    size_train_T = numel(train_T);
    size_train_F = numel(train_F);
    if size_train_T > size_train_F
        n = fix(size_train_T/size_train_F);
        m = mod(size_train_T,size_train_F);
        r = randperm(size_train_F, m);
        train_F0 = train_F;
        for k = 1:n-1
            train_F = [train_F train_F0];
        end
        for k = 1:m
            train_F = [train_F train_F0(r(k))];
        end
    elseif size_train_T < size_train_F
        n = fix(size_train_F/size_train_T);
        m = mod(size_train_F,size_train_T);
        r = randperm(size_train_T, m);
        train_T0 = train_T;
        for k = 1:n-1
            train_T = [train_T train_T0];
        end
        for k = 1:m
            train_T = [train_T train_T0(r(k))];
        end
    end
    copyed_T = train_T;
    copyed_F = train_F;
end