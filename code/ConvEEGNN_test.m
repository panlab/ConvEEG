function [opts] = ConvEEGNN_test(net, x, y, opts)
    %  feedforward
    for i = 1 : opts.testepochs
        net = ConvEEGNN_ff(net, x{i});
        opts.output(i,:) = net.o;
        
        if net.o(1) == net.o(2)
            bad = 1;
        else            
            [~, h] = max(net.o);
            [~, a] = max(y(i,:));
            bad = find(h ~= a); 
            if bad == 1
                opts.erroridx = [opts.erroridx i];
            end
        end
        opts.error = opts.error + numel(bad);
    end
    
    opts.errors = opts.error;
    opts.error = opts.error / opts.testepochs;
end