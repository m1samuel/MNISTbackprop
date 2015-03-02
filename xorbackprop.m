function [W_in,W_out] = xorbackprop(f,h,lr)
    W_in = (-2/sqrt(3)).*rand(3,h) + 1/sqrt(3);
    W_out = (-2/sqrt(3)).*rand(h+1,1) + 1/sqrt(3);
    In = [-1 -1; -1 1; 1 -1; 1 1];
    In_Out = [(1/std(In(1:end,1))*(In(1:end,1) - mean(In(1:end,1)))) (1/std(In(1:end,2))*(In(1:end,2) - mean(In(1:end,2))))];
    syms u;
    f_prime =  matlabFunction(diff(f(u)));
    max_epochs = 200000;
    error_threshold = 0.2;
    error = 1;
    epoch = 0;
    In_Out = [In_Out ones(4,1)]; %put the biasis into In_Out
    In_Out = [In_Out [-1; 1; 1; -1]]; %set targets
    deltaw_in = zeros(3,h);
    deltaw_out = zeros(h+1,1);
    H = [zeros(h,1); 1];
    ec = 0;
    In_Out = In_Out(randperm(size(In_Out,1)),:);
    while ((epoch < max_epochs) && (error > error_threshold))
        for i=1:length(In_Out)
            %compute Hidden unit values
            for j=1:h
                H(j) = f(In_Out(i,1:end-1) * W_in(1:end,j));
            end
            %randomly update weights for hidden to output layer (bias inc)
            sumwikdeltak = 0;
            k = randi(numel(W_out));
            deltak = (In_Out(i,end) - f(H'*W_out)) * f_prime(H'*W_out);
            deltaw = lr*deltak*H(k) + 0.9 * deltaw_out(k);
            W_out(k) = W_out(k) + deltaw;
            sumwikdeltak = sumwikdeltak + deltak * W_out(k);
            deltaw_out(k) = deltaw;
            %randomly update weights for input to hidden layer (bias inc)
            k = randi(3);
            j = randi(h);
            deltaj = f_prime(In_Out(i,1:end-1) * W_in(1:end,j)) * sumwikdeltak;
            deltaw = lr* deltaj * In_Out(i,k) + 0.9 * deltaw_in(k,j);
            W_in(k,j) = W_in(k,j) + deltaw;
            deltaw_in(k,j) = deltaw;
        end
        %compute error
        for k=1:length(In_Out)
            for j=1:h
                H(j) = f(round(In_Out(k,1:end-1)) * W_in(1:end,j));
            end
            O = f(W_out' * H);
            if (O >= 0)
                O = 1;
            else
                O = -1;
            end
            if (((In_Out(k,end) == 1) && ~(O == 1)) || ((In_Out(k,end) == -1) && ~(O == -1)))
                ec = ec + 1;
            end
        end
        error = ec/4
        ec = 0;
        epoch = epoch + 1
    end
end

