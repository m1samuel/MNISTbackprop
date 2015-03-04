function [W_in,W_out] = MNISTbackprop(h,lr,images,labels,bs)
    In_Out = images';
    In_Out = [In_Out ones(length(In_Out),1) labels];
    In_Out = In_Out(randperm(size(In_Out,1)),:);
    W_in = (-2/sqrt(785)).*rand(785,h) + 1/sqrt(785);
    W_out = (-2/sqrt(785)).*rand(10,h+1) + 1/sqrt(785);
    deltaw_in = zeros(785,h);
    deltaw_out = zeros(10,h+1);
    f = @(u) exp(u)./sum(exp(u));
    H = [zeros(h,1); 1];
    max_epochs = 10000;
    error_threshold = .20;
    error = 1;
    epoch = 0;
    ec = 0;
    batchsize = bs;
    errvec = 1;
    epochvec = 0;
    sameerrorcount = 0;
    prev_error = 0;
    while ((epoch < max_epochs) && (error > error_threshold))
        i = randi(batchsize);
        %compute Hidden unit values
        H = f([In_Out(i,1:end-1) * W_in(1:end,:) 1]');
        %compute output
        Output = f((W_out*H));
        %randomly update weights for hidden to output layer (bias inc)
        t = zeros(10,1);
        t(In_Out(i,end)+1) = 1;
        %compute delta k's for output layer
        deltak = t - Output;
        %compute delta j's for output layer
        deltaj = H(1:end-1).*(1-H(1:end-1)).*(W_out(:,1:end-1)' * deltak);
        %update weights from hidden units to output units
        deltaw =  lr*deltak*H' + 0.9 * deltaw_out;
        W_out = W_out + deltaw;
        deltaw_out = deltaw;
        %update weights from input units to hidden units
        deltaw = (lr*deltaj*In_Out(i,1:end-1))' + 0.9 * deltaw_in;
        W_in = W_in + deltaw;
        deltaw_in = deltaw;
        %compute error
        for j=1:batchsize
            In_Out(j,end)+1;
            H = f([In_Out(j,1:end-1) * W_in(1:end,:) 1]');
            Output = f((W_out*H));
            [~,i] = max(Output);
            if (~isequal(i,In_Out(j,end)+1))
                ec = ec + 1;
            end
        end
        error = ec/batchsize
        ec = 0;
        epoch = epoch + 1
                if (prev_error == error)
            sameerrorcount = sameerrorcount + 1;
        else
            sameerrorcount = 0;
        end
        if (sameerrorcount == 500)
            W_in = (-2/sqrt(785)).*rand(785,h) + 1/sqrt(785);
            W_out = (-2/sqrt(785)).*rand(10,h+1) + 1/sqrt(785);
            sameerrorcount = 0;
        end
        prev_error = error;
        errvec = [errvec; error];
        epochvec = [epochvec; epoch];
    end
    plot(epochvec,errvec)
end

