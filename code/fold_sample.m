function [test_x_all,test_y_all,train_x_all,train_y_all,valid_x_all,valid_y_all,...
    Ttest_all,Tvalid_all,opts_test_all,opts_valid_all,opts_train_all] = ...
    fold_sample(all_T,all_F,SemanticRecollected_d,SemanticNonRecollected_d)
%% load data
size_T = numel(all_T);
size_F = numel(all_F);
%% sample size for test, train, valid
% test sample
opts_test_stepT = 1;   %test basic step
opts_test_stepF = 1;
opts_test.testepochs_T = opts_test_stepT;
opts_test.testepochs_F = opts_test_stepF;
opts_test.testepochs = opts_test.testepochs_T + opts_test.testepochs_F;
% valid sample 30%
valid = 0.3;
opts_valid.testepochs = floor(min((size_T-opts_test.testepochs_T),(size_F-opts_test.testepochs_F))*valid)*2;
opts_valid.testepochs_T = opts_valid.testepochs/2;
opts_valid.testepochs_F = opts_valid.testepochs_T;
%train sample 70%
opts_train.trainepochs = max((size_T-opts_test.testepochs_T-opts_valid.testepochs_T),...
    (size_F-opts_test.testepochs_F-opts_valid.testepochs_F))*2;
opts_train.trainepochs_T = opts_train.trainepochs/2;
opts_train.trainepochs_F = opts_train.trainepochs_T;
% label for true(remember)
Ttest = 1:opts_test.testepochs_T;
Tvalid = 1:2:opts_valid.testepochs;
%% N
N = min(floor(size_T/opts_test.testepochs_T),floor(size_F/opts_test.testepochs_F));

test_x = cell(1,opts_test.testepochs);
test_y = zeros(opts_test.testepochs,2);
train_x = cell(1,opts_train.trainepochs);
train_y = zeros(opts_train.trainepochs,2);
valid_x = cell(1,opts_valid.testepochs);
valid_y = zeros(opts_valid.testepochs,2);
for i = 1:N
    % sample size for test, train, valid
    if i == N
        % test sample
        opts_test.testepochs_T = opts_test_stepT + (size_T - N*opts_test_stepT);
        opts_test.testepochs_F = opts_test_stepF + (size_F - N*opts_test_stepF);
        opts_test.testepochs = opts_test.testepochs_T + opts_test.testepochs_F;
        % valid sample 30%
        opts_valid.testepochs = floor(min((size_T-opts_test.testepochs_T),(size_F-opts_test.testepochs_F))*valid)*2;
        opts_valid.testepochs_T = opts_valid.testepochs/2;
        opts_valid.testepochs_F = opts_valid.testepochs_T;
        %train sample 70%
        opts_train.trainepochs = max((size_T-opts_test.testepochs_T-opts_valid.testepochs_T),...
            (size_F-opts_test.testepochs_F-opts_valid.testepochs_F))*2;
        opts_train.trainepochs_T = opts_train.trainepochs/2;
        opts_train.trainepochs_F = opts_train.trainepochs_T;
        % label for true(remember)
        Ttest = 1:opts_test.testepochs_T;
        Tvalid = 1:2:opts_valid.testepochs;
        
        test_x = cell(1,opts_test.testepochs);
        test_y = zeros(opts_test.testepochs,2);
        train_x = cell(1,opts_train.trainepochs);
        train_y = zeros(opts_train.trainepochs,2);
        valid_x = cell(1,opts_valid.testepochs);
        valid_y = zeros(opts_valid.testepochs,2);
    end
    
    test_T_range = (opts_test_stepT*(i-1)+1) : (opts_test_stepT*(i-1) + opts_test.testepochs_T);
    test_F_range = (opts_test_stepF*(i-1)+1) : (opts_test_stepF*(i-1) + opts_test.testepochs_F);
    % test sample T
    for j = 1:opts_test.testepochs_T
        test_x{j} = SemanticRecollected_d{test_T_range(j)};
        test_y(j,:) = [0 1];
    end
    valid_T_range = (test_T_range(j)+1:test_T_range(j)+opts_valid.testepochs_T);
    if any(valid_T_range>size_T)
        index_out_T = find(valid_T_range>size_T);
        valid_T_range(index_out_T) = mod(valid_T_range(index_out_T),size_T);
    end
    % test sample F
    for j = 1:opts_test.testepochs_F
        test_x{j+opts_test.testepochs_T} = SemanticNonRecollected_d{test_F_range(j)};
        test_y(j+opts_test.testepochs_T,:) = [1 0];
    end
    valid_F_range = (test_F_range(j)+1:test_F_range(j)+opts_valid.testepochs_F);
    if any(valid_F_range>size_F)
        index_out_F = find(valid_F_range>size_F);
        valid_F_range(index_out_F) = mod(valid_F_range(index_out_F),size_F);
    end
    
    % valid samlpe
    for j = 1:opts_valid.testepochs/2
        valid_x{2*j-1} = SemanticRecollected_d{valid_T_range(j)};
        valid_y(2*j-1,:) = [0 1];
        valid_x{2*j} = SemanticNonRecollected_d{valid_F_range(j)};
        valid_y(2*j,:) = [1 0];
    end
    % train sample
    train_T_range = setdiff(all_T,[test_T_range,valid_T_range]);
    train_F_range = setdiff(all_F,[test_F_range,valid_F_range]);
    [train_T_range,train_F_range] = copySmallSample(train_T_range,train_F_range);
    for j = 1:opts_train.trainepochs/2
        train_x{2*j-1} = SemanticRecollected_d{train_T_range(j)};
        train_y(2*j-1,:) = [0 1];
        train_x{2*j} = SemanticNonRecollected_d{train_F_range(j)};
        train_y(2*j,:) = [1 0];
    end
    test_x_all{i} = test_x;
    test_y_all{i} = test_y;
    train_x_all{i} = train_x;
    train_y_all{i} = train_y;
    valid_x_all{i} = valid_x;
    valid_y_all{i} = valid_y;
    Ttest_all{i} = Ttest;
    Tvalid_all{i} = Tvalid;
    opts_test_all{i} = opts_test;
    opts_valid_all{i} = opts_valid;
    opts_train_all{i} = opts_train;
end