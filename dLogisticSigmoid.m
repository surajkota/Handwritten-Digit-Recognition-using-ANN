function y = dLogisticSigmoid(x)
% dLogisticSigmoid Derivative of the logistic sigmoid.
% 
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector where the derivative of the logistic sigmoid was
% applied element by element.
%
        temp = 1./(1 + exp(-x));
    y = temp.*(1-temp);
    %logisticSigmoid(x).*(1 - logisticSigmoid(x));
end