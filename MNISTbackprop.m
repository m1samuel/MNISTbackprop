function [W_in,W_out] = MNISTbackprop(h,lr,images,labels)
    In_Out = images';
    In_Out = [In_Out ones(length(In_Out),1) labels];
    In_Out = In_Out(randperm(size(In_Out,1)),:);
    W_in = (-2/sqrt(785)).*rand(785,h) + 1/sqrt(785);
    W_out = (-2/sqrt(785)).*rand(10,h+1) + 1/sqrt(785);
    deltaw_in = zeros(785,h);
    deltaw_out = zeros(10,h+1);
    H = [zeros(h,1); 1];
    max_epochs = 200000;
    error_threshold = .252;
    error = 1;
    epoch = 0;
    ec = 0;
    batchsize = 20;
    while ((epoch < max_epochs) && (error > error_threshold))
        for i=1:batchsize
            %compute Hidden unit values
            H = softmax([In_Out(i,1:end-1) * W_in(1:end,:) 1]');
            %compute output
            Output = softmax((W_out*H));
            %randomly update weights for hidden to output layer (bias inc)
            t = zeros(10,1);
            t(In_Out(i,end)+1) = 1;
            %compute delta k's for output layer
            deltak = t - Output;
            %update weights from hidden units to output units
            for j=randsample(randperm(h+1),8)
                deltaw =  lr*deltak*H(j) + 0.9 * deltaw_out(1:end,j);
                W_out(1:end,j) = W_out(1:end,j) + deltaw;
                deltaw_out(1:end,j) = deltaw;
            end
            deltaj = H(1:end-1).*(1-H(1:end-1)).*(W_out(:,1:end-1)' * deltak);
            %update weights from input units to output units
            for j=randsample(randperm(785),222)
                deltaw = (lr*deltaj*In_Out(i,j))' + 0.9 * deltaw_in(j,1:end);
                W_in(j,1:end) = W_in(j,1:end) + deltaw;
                deltaw_in(j,1:end) = deltaw;
            end
        end
        %compute error
        for j=1:batchsize
            In_Out(j,end)+1
            H = softmax([In_Out(j,1:end-1) * W_in(1:end,:) 1]');
            Output = softmax((W_out*H));
            t = zeros(10,1);
            t(In_Out(j,end)+1) = 1;
            [m i] = max(Output);
            if (~isequal(i,In_Out(j,end)+1))
                ec = ec + 1;
            end
        end
        error = ec/batchsize
        ec = 0;
        epoch = epoch + 1
    end
end

