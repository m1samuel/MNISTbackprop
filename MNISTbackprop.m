function [W_in,W_out] = MNISTbackprop(g,h,lr,images,labels,bs)
    % --------------------- Input Params ------------------------------
    % g is the hidden unit's activation function
    % h is the number of hidden units to use
    % lr is the learning rate (need to implement adaptive learning rate)
    % images is a k by n matrix of handwritten digits where n is the number 
    %      of training images and k is the number of pixels each image has.
    % labels is n by 1 vector of known values for the handwritten digits
    % bs is the batch size.
    %
    % ------------------------ Output ---------------------------------
    % W_in is a matrix of weights from the input layer to the hidden layer
    % W_out is a matrix of weights from the hidden layer to the output
    % layer.
    % -----------------------------------------------------------------
    In_Out = images';
    In_Out = [In_Out ones(length(In_Out),1) labels];
    W_in = (-2/sqrt(785)).*rand(785,h) + 1/sqrt(785);
    W_out = (-2/sqrt(785)).*rand(10,h+1) + 1/sqrt(785);
    syms u;
    g_prime = matlabFunction(diff(g(u)));
    max_epochs = 10;
    error_threshold = .005;
    error = 1;
    epoch = 0;
    ec = 0;
    batchsize = bs;
    errvec = 1;
    epochvec = 0;
    while ((epoch < max_epochs) && (error > error_threshold))
        deltaw_in = zeros(785,h);
        deltaw_out = zeros(10,h+1);
        %i = randi(batchsize);
        In_Out = In_Out(randperm(size(In_Out,1)),:);
        for i = 1:batchsize
            %compute Hidden unit values
            H = g([In_Out(i,1:end-1) * W_in(1:end,:) 1]');
            %compute output
            Output = softmax((W_out*H));
            t = zeros(10,1);
            t(In_Out(i,end)+1) = 1;
            %compute deltas for output layer
            deltak = t - Output;
            %compute deltas for hidden layer
            deltaj = g_prime(H(1:end-1)).*(W_out(:,1:end-1)' * deltak);
            %update weights from hidden units to output units
            deltaw_out =  lr*(deltak*H' + 0.9 * deltaw_out);
            W_out = W_out + deltaw_out;
            %update weights from input units to hidden units
            deltaw_in = lr*((deltaj*In_Out(i,1:end-1))' + 0.9 * deltaw_in);
            W_in = W_in + deltaw_in;
        end
        %compute error
        for j=1:batchsize
            H = g([In_Out(j,1:end-1) * W_in(1:end,:) 1]');
            Output = softmax((W_out*H));
            [~,i] = max(Output);
            if (~isequal(i,In_Out(j,end)+1))
                ec = ec + 1;
            end
        end
        error = ec/batchsize
        ec = 0;
        epoch = epoch + 1
        errvec = [errvec; error];
        epochvec = [epochvec; epoch];
    end
    plot(epochvec,errvec)
end