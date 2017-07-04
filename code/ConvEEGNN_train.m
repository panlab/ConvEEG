function net = ConvEEGNN_train(net, x, y, opts)
for i = 1 : opts.trainepochs
    net = ConvEEGNN_ff(net, x{i});
    net = ConvEEGNN_bp(net, y(i,:));     
    net = ConvEEGNN_applygrads(net);
end
end