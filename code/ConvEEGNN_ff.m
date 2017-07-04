function net = ConvEEGNN_ff(net, x)
%% Convolution layer
net.layers{1}.a{1} = x; 
net.layers{2}.a = mat2cell(1.7159*tanh(2/3*(cell2mat(net.layers{2}.k')*net.layers{1}.a{1}...
    +repmat(cell2mat(net.layers{2}.b'),1,net.layers{2}.perNeural))),ones(1,net.layers{2}.outputmaps),net.layers{2}.perNeural)';
%% Convolution-Subsampling layer
inputmaps = net.layers{2}.outputmaps;
T = net.layers{3}.outputmaps/net.layers{2}.outputmaps;
for i = 1 : inputmaps
    w = reshape(net.layers{2}.a{i}', max(net.layers{3}.kernelsize), net.layers{3}.perNeural)';
    net.layers{3}.a(T*(i-1)+1:T*i) = mat2cell(1.7159*tanh(2/3*(w*cell2mat(net.layers{3}.k{i})...
        +repmat(cell2mat(net.layers{3}.b{i}),net.layers{3}.perNeural,1)))',ones(1,T),net.layers{3}.perNeural);
end    
%% Hidden layer
csa = cell2mat(net.layers{3}.a);    
hk = cell2mat(net.layers{4}.k');    
z = csa*hk';
net.layers{4}.a = sigm(z + (net.layers{4}.b)');
%% Output layer
z = net.layers{4}.a*(net.ffW)';
net.o = sigm(z + net.ffb');
end