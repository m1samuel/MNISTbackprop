function [W_in,W_out] = xorbackprop(f,h,lr)
    In = [-1 -1; -1 1; 1 -1; 1 1];
    In_Out = [(1/std(In(1:end,1))*(In(1:end,1) - mean(In(1:end,1)))) (1/std(In(1:end,2))*(In(1:end,2) - mean(In(1:end,2))))];
    syms u;
    f_prime =  matlabFunction(diff(f(u)));
    max_epochs = 2000;
    error_threshold = 0;
    In_Out = [In_Out ones(4,1)]; %put the biasis into In_Out
    In_Out = [In_Out [-1; 1; 1; -1]]; %set targets
    W_in = (-2/sqrt(3)).*rand(3,h) + 1/sqrt(3);
    W_out = (-2/sqrt(3)).*rand(h+1,1) + 1/sqrt(3);
    error = 1;
    epoch = 0;
    ec = 0;
    errvec = 1;
    epochvec = 0;
    while ((epoch < max_epochs) && (error > error_threshold))
        In_Out = In_Out(randperm(size(In_Out,1)),:);
        deltaw_in = zeros(3,h);
        deltaw_out = zeros(h+1,1);
        for i=1:4
            %compute Hidden unit values
            H = [f(In_Out(i,1:end-1) * W_in(1:end,:)) 1]';
            %compute output
            Output = f(H'*W_out);
            %compute delta for output layer
            deltak = (In_Out(i,end) - Output) * f_prime(H'*W_out);
            %compute deltas for hidden layers
            deltaj = f_prime(In_Out(i,1:end-1) * W_in(1:end,:)) * (sum(deltak * W_out(1:end-1)));
            %update weights for hidden to output layer
            deltaw_out = lr * (deltak*H + 0.9 * deltaw_out);
            W_out = W_out + deltaw_out;
            %update weights for input to hidden layer
            deltaw_in = lr * (In_Out(i,1:3)' * deltaj + 0.9 * deltaw_in);
            W_in = W_in + deltaw_in;
        end
        %compute error
        for i=1:length(In_Out)
            H = [f(In_Out(i,1:end-1) * W_in(1:end,:)) 1]';
            O = f(W_out' * H);
            if (O >= 0)
                O = 1;
            else
                O = -1;
            end
            if (((In_Out(i,end) == 1) && ~(O == 1)) || ((In_Out(i,end) == -1) && ~(O == -1)))
                ec = ec + 1;
            end
        end
        error = ec/4
        ec = 0;
        epoch = epoch + 1
        errvec = [errvec; error];
        epochvec = [epochvec; epoch];
    end
    plot(epochvec,errvec)
end