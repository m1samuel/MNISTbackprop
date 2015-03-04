function [e,imlabmissed] = testMNIST(images,labels,W_in,W_out)
    In_Out = [images' ones(length(images'),1) labels];
    ec = 0;
    f = @(u)exp(u)./sum(exp(u));
    misclassified = zeros(10,11);
    % misclassified is a matrix where the first column is
    % what the digit really was and the corresponding rows
    % to each element in the column is what our net though it was
    imlabmissed = zeros(10000,785);
    for i=1:length(images)
        %compute Hidden unit values
        H = f([In_Out(i,1:end-1) * W_in(1:end,:) 1]');
        %compute output
        Output = f((W_out*H));
        [~,j] = max(Output);
        if (~isequal(j,In_Out(i,end)+1))
            imlabmissed(i,:) = [In_Out(i,1:end-2) In_Out(i,end)];
            misclassified(In_Out(i,end)+1,1) = misclassified(In_Out(i,end)+1)+1;
            misclassified(In_Out(i,end)+1,j+1) = misclassified(In_Out(i,end)+1,j+1)+1;
            ec = ec + 1;
        end
    end
    misclassified
    e = ec/10000
end

