function net = ConvEEGNN_bp(net, y)
net.e = net.o - y;  %error
net.L = 1/2 * sum(net.e(:) .^ 2);
wk = cell2mat(net.layers{4}.k'); 
csk = [net.layers{3}.a{:}]; 
%% output layer
net.od = -net.e .* (net.o .* (1 - net.o)); % bp sensitivity (sigmoid)
%% hidden layer
net.layers{4}.d = net.layers{4}.a .* (1 - net.layers{4}.a).* (net.od*net.ffW);% hidden sensitivity (sigmoid)
%% convolution & subsampling layer
net.fvd = 0;
net.fvd = net.layers{4}.d * wk; 
t = 2/3 *(1.7159 - cell2mat(net.layers{3}.a).* cell2mat(net.layers{3}.a)/1.7159);
net.layers{3}.d = mat2cell( t.*net.fvd,1,net.layers{3}.perNeural*ones(1, net.layers{3}.outputmaps)); % cs sensitivity with reshape (tanh)
%% convolution layer
T = net.layers{3}.outputmaps/net.layers{2}.outputmaps;
for i = 1 : net.layers{2}.outputmaps    
    net.layers{2}.d{i} = 0;
    t = 2/3 *(1.7159 - net.layers{2}.a{i}.* net.layers{2}.a{i}/1.7159);   %(tanh)
    up = net.layers{3}.d(1,(i-1)*T+1:i*T);
    up1 = 0;
    for j = 1:T
       tmp = ones(net.layers{3}.kernelsize(1),1)*up{1,j};
       tmp = tmp(:);
       tmp = tmp(1:(net.layers{2}.perNeural - net.layers{3}.kernelsize(1)+1));
       up1 = up1 + convn(tmp,rot180(net.layers{3}.k{i}{j}),'full');
    end
    net.layers{2}.d{i} = t.*up1'; % c sensitivity
end
%% -----------------calc gradients--------------------------------
%% convolution layer
net.layers{2}.dk = mat2cell(net.layers{2}.learningRate *net.layers{1}.a{1}...
    *reshape([net.layers{2}.d{:}],net.layers{2,1}.perNeural,net.layers{2}.outputmaps)...
    ,size(net.layers{1}.a{1},1),ones(1,net.layers{2}.outputmaps));
net.layers{2}.db = num2cell(net.layers{2}.learningRate * ...
    sum(reshape([net.layers{2}.d{:}],net.layers{2,1}.perNeural,net.layers{2}.outputmaps),1));
%% convolution & subsampling layer
for i = 1 : net.layers{2}.outputmaps
    up = net.layers{3}.d(1,(i-1)*T+1:i*T);
    wd = reshape(net.layers{2}.a{i},net.layers{3}.kernelsize(1),net.layers{3,1}.perNeural);
    wd = wd*reshape([up{:}],net.layers{3,1}.perNeural,T);
    net.layers{3}.dk{i} = mat2cell(net.layers{3}.learningRate * wd,net.layers{3}.kernelsize(1),ones(1,T));
    net.layers{3}.db{i} = num2cell(net.layers{3}.learningRate * sum(reshape([up{:}],net.layers{3,1}.perNeural,T),1));
end
%% hidden layer
up = net.layers{4}.learningRate * csk' * net.layers{4}.d;
up = up(:)';
net.layers{4}.dk = mat2cell(up,1,(net.layers{3}.perNeural*net.layers{3}.outputmaps)*ones(1,net.layers{4}.perNeural));%CHANGE
net.layers{4}.db = net.layers{4}.learningRate * net.layers{4}.d;
%% output layer
net.dffW = net.learningRate * (net.od)' * net.layers{4}.a;
net.dffb = net.learningRate * net.od;
end

function X = rot180(X)
X = flipdim(flipdim(X, 1), 2);
end
