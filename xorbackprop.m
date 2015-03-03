function [W_in,W_out] = xorbackprop(f,h,lr)
    In = [-1 -1; -1 1; 1 -1; 1 1];
    In_Out = [(1/std(In(1:end,1))*(In(1:end,1) - mean(In(1:end,1)))) (1/std(In(1:end,2))*(In(1:end,2) - mean(In(1:end,2))))];
    syms u;
    f_prime =  matlabFunction(diff(f(u)));
    max_epochs = 2000;
    error_threshold = 0.2;
    In_Out = [In_Out ones(4,1)]; %put the biasis into In_Out
    In_Out = [In_Out [-1; 1; 1; -1]]; %set targets
    W_in = (-2/sqrt(3)).*rand(3,h) + 1/sqrt(3);
    W_out = (-2/sqrt(3)).*rand(h+1,1) + 1/sqrt(3);
    error = 1;
    epoch = 0;
    deltaw_in = zeros(3,h);
    deltaw_out = zeros(h+1,1);
    ec = 0;
    In_Out = In_Out(randperm(size(In_Out,1)),:);
    errvec = 1;
    epochvec = 0;
    sameerrorcount = 0;
    prev_error = 0;
    while ((epoch < max_epochs) && (error > error_threshold))
        i = randi(4);
        %compute Hidden unit values
        H = [f(In_Out(i,1:end-1) * W_in(1:end,:)) 1]';
        %compute output
        Output = f(H'*W_out);
        %compute delta k for output layer
        deltak = (In_Out(i,end) - Output) * f_prime(H'*W_out);
        %compute delta j's for hidden layers
        deltaj = f_prime(In_Out(i,1:end-1) * W_in(1:end,:)) * (sum(deltak * W_out(1:end-1)));
        %update weights for hidden to output layer
        deltaw = lr*deltak*H + 0.9 * deltaw_out;
        W_out = W_out + deltaw;
        deltaw_out = deltaw;
        %update weights for input to hidden layer
        deltaw = lr * In_Out(i,1:3)' * deltaj + 0.9 * deltaw_in;
        W_in = W_in + deltaw;
        deltaw_in = deltaw;
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
        if (prev_error == error)
            sameerrorcount = sameerrorcount + 1;
        else
            sameerrorcount = 0;
        end
        if (sameerrorcount == 100)
            W_in = (-2/sqrt(3)).*rand(3,h) + 1/sqrt(3);
            W_out = (-2/sqrt(3)).*rand(h+1,1) + 1/sqrt(3);
            sameerrorcount = 0;
        end
        prev_error = error;
        errvec = [errvec; error];
        epochvec = [epochvec; epoch];
    end
    plot(epochvec,errvec)
end

