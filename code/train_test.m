function [errors_size,FNs_size,cnn_selected,label_y,output_y] = ...
    train_test(peroid,cskernel,train_x, train_y, opts_train,valid_x,valid_y,opts_valid,test_x,test_y,opts_test,Ttest)
%% construct a convolutional neural network
m = 0.2;
ckernelsize = 30;
cmaps = 1;
cneuron = length(peroid);
cskernelsize = cskernel;
csmaps = 1;
csneuron = 3;  %cneuron/cskernelsize;
hiddenNeuron = 10;
learningRate1 = 2*m/(cneuron*sqrt(ckernelsize));
learningRate2 = 2*m/(csneuron*sqrt(cskernelsize));
learningRate3 = 2*m/(sqrt(csneuron*cmaps*csmaps));
learningRate4 = 2*m/(sqrt(hiddenNeuron));
cnn.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', cmaps, 'kernelsize', [1 ckernelsize], 'perNeural', cneuron, 'learningRate', learningRate1)     %convolution layer
    struct('type', 'cs', 'outputmaps', csmaps*cmaps, 'kernelsize', [cskernelsize 1], 'perNeural', csneuron, 'learningRate', learningRate2)     %convolution and subsampling layer
    struct('type', 'h', 'outputmaps', 1, 'perNeural', hiddenNeuron, 'learningRate', learningRate3)                  % hidden layer
    };
cnn.learningRate = learningRate4;          %output layer
cnn.perNeural = 2;

%setup a cnn
cnn_N = ConvEEGNN_setup(cnn);
%iterator
iter = 200;
error_valid = zeros(1,iter);
cnn_pre = cell(1,iter);
for epoch = 1:iter
    % train
    cnn_N = ConvEEGNN_train(cnn_N, train_x, train_y, opts_train);
    cnn_pre{epoch} = cnn_N;
    %valid
    opts_valid.error = 0;
    opts_valid.erroridx = [];
    opts_valid.output = [];
    opts_valid.FN = 0;
    opts_valid.errors = 0;
    [opts_valid] = ConvEEGNN_test(cnn_N,valid_x,valid_y,opts_valid);
    
    error_valid(epoch) = opts_valid.error;
end
[~,index_min_er] = min(error_valid);
cnn_selected = cnn_pre{index_min_er};

opts_test.error = 0;
opts_test.erroridx = [];
opts_test.output = [];
opts_test.FN = 0;
opts_test.errors = 0;
[opts_test] = ConvEEGNN_test(cnn_selected,test_x,test_y,opts_test);
opts_test.FN = intersect(Ttest,opts_test.erroridx);

errors_size = opts_test.errors;
FNs_size = numel(opts_test.FN);

label_y = test_y(:,2);
output_y = opts_test.output;

end