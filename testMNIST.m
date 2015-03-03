function e = testMNIST(images,labels,W_in,W_out)
    In_Out = images';
    In_Out = [In_Out ones(length(In_Out),1) labels];
    ec = 0;
    f = @(u)exp(u)./sum(exp(u));
    misclassified = zeros(10,1);
    for i=1:length(images)
        %compute Hidden unit values
        H = f([In_Out(i,1:end-1) * W_in(1:end,:) 1]');
        %compute output
        Output = f((W_out*H));
        [~,j] = max(Output);
        misclassified(In_Out(i,end)+1) = misclassified(In_Out(i,end)+1) +1;
        if (~isequal(j,In_Out(i,end)+1))
            ec = ec + 1;
        end
    end
    misclassified
    e = ec/10000
end

