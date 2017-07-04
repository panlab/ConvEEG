function [accuracy,FN,testp,cnn] = ConvEEGNN(peroid,SemanticRecollected_d,SemanticNonRecollected_d,all_T,all_F,cskernel)
%% train
[test_x,test_y,train_x,train_y,valid_x,valid_y,Ttest,~,opts_test,opts_valid,opts_train] = ...
fold_sample(all_T,all_F,SemanticRecollected_d,SemanticNonRecollected_d);
N = size(test_x,2);
cskernel = mat2cell(ones(1,N)*cskernel,1,ones(1,N));  
FNs_size = zeros(1,1);
errors_size = zeros(1,1);
cnn = cell(1,1);

parfor iN = 1:N    
    [errors_size(iN),FNs_size(iN),cnn{iN},label_y{iN,1},output_y{iN,1}] = train_test(peroid,...
        cskernel{iN},train_x{iN}, train_y{iN}, opts_train{iN},valid_x{iN},valid_y{iN},opts_valid{iN},...
        test_x{iN},test_y{iN},opts_test{iN},Ttest{iN});   
end
size_T = numel(all_T);
size_F = numel(all_F);
accuracy = 1 - sum(errors_size)/(size_T + size_F);
FN = sum(FNs_size)/sum(errors_size);
x = fsolve(@(Z)(0.5*(size_T+size_F)/(size_T+size_F+Z^2) + 1/2 * Z^2 /(size_T+size_F+Z^2) ...
    + Z * sqrt(1/(size_T+size_F+Z^2) * (0.5*0.5*(size_T+size_F)/(size_T+size_F+Z^2) ...
    + 1/4*Z^2 /(size_T+size_F+Z^2))) - accuracy),0);
testp = 2*(1-normcdf(abs(x),0,1));
end
