clear;
inputValues1 = loadMNISTImages('train-images.idx3-ubyte');
labels1 = loadMNISTLabels('train-labels.idx1-ubyte');
inputTesting1 = loadMNISTImages('t10k-images.idx3-ubyte');

Testlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
inputValues=[];
inputValidation=[];
inputTesting=[];
labels=labels1(1:50000,:);
for i=1:784
    if mod(i,2)==0
      inputValues=[inputValues; inputValues1(i,:)];
      inputValidation=[inputValidation; inputValues1(i,50001:60000)];
      inputTesting=[inputTesting; inputTesting1(i,:)];
    end
end

Validationlabels=labels1(50001:60000,:);
% Transform the labels to correct target values.
targetValues = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    targetValues(labels(n) + 1, n) = 1;
end;

% Transform the labels to correct target values for validation.
targetValuesValidation = 0.*ones(10, size(Validationlabels, 1));
for n = 1: size(Validationlabels, 1)
    targetValuesValidation(Validationlabels(n) + 1, n) = 1;
end;

% Choose form of MLP:
numberOfHiddenUnits = 80;

% Choose appropriate parameters.
learningRate = 0.01;

% Choose activation function.
activationFunction = @RELU1;
dActivationFunction = @dRELU1;

% Choose batch size and epochs. Remember there are 60k input values.
batchSize = 50000;
epochs = 14;
E=1;
errorV = 0;
fprintf('Train twolayer perceptron with %d hidden units.\n', numberOfHiddenUnits);
fprintf('Learning rate: %d.\n', learningRate);

trainingSetSize = size(inputValues, 2);

% Input vector has 784 dimensions.
inputDimensions = size(inputValues, 1);
% We have to distinguish 10 digits.
outputDimensions = size(targetValues, 1);

% Initialize the weights for the hidden layer and the output layer.
hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
outputWeights = rand(outputDimensions, numberOfHiddenUnits);

hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
outputWeights = outputWeights./size(outputWeights, 2);

% n = zeros(batchSize);

figure; hold on;
count=0;
previous_outputdeltaw=zeros(outputDimensions, numberOfHiddenUnits);
previous_hiddendeltaw=zeros(numberOfHiddenUnits, inputDimensions);
alpha=0.8;

for i=1:epochs

    for k = 1: batchSize
        % Select which input vector to train on.
        %n(k) = k;%floor(rand(1)*trainingSetSize + 1);
        
        % Propagate the input vector through the network.
        inputVector = inputValues(:, k);
        hiddenActualInput = hiddenWeights*inputVector;
        hiddenOutputVector = activationFunction(hiddenActualInput);
        outputActualInput = outputWeights*hiddenOutputVector;
        outputVector = activationFunction(outputActualInput);
        
        targetVector = targetValues(:, k);
        
        % Backpropagate the errors.
        outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
        hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);
        
        presentOutputDelta = (learningRate.*outputDelta*hiddenOutputVector') - (alpha.*previous_outputdeltaw);
        presentHiddenDelta = (learningRate.*hiddenDelta*inputVector') - (alpha.*previous_hiddendeltaw);
        outputWeights = outputWeights - presentOutputDelta;
        hiddenWeights = hiddenWeights - presentHiddenDelta;
        previous_outputdeltaw=presentOutputDelta;
        previous_hiddendeltaw=presentHiddenDelta;
        
        if mod(k,1000)==0
            prob=rand;
            if(prob<0.5)
                layer=rand;
                if(rand<0.5)
                hiddenWeights(:,randi([1,inputDimensions]))=0;
                else
                outputWeights(:,randi([1,numberOfHiddenUnits]))=0;
                end
            end
            fprintf('epoch %d', i);
            fprintf('At %d\n', k);
        end
    end;

    % Calculate the error for plotting.
    count=count+1;
    error = 0;
    classificationErrorsTr = 0;
    correctlyClassifiedTr = 0;
    for k1 = 1: batchSize
        inputVector = inputValues(:, k1);
        targetVector = targetValues(:,k1);
        outputVectorTr=activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector));
        error = error + norm(outputVectorTr - targetVector, 2);
        max = -100;
        classTr((k1)) = 1;
        for ii = 1: size(outputVector, 1)
            if outputVectorTr(ii) > max
                max = outputVectorTr(ii);
                classTr((k1)) = ii-1;
            end;
        end;
        
        if classTr(k1) == labels((k1))
            correctlyClassifiedTr = correctlyClassifiedTr + 1;
        else
            classificationErrorsTr = classificationErrorsTr + 1;
        end;
        
    end;
    classification_accuracyTr(count)=(correctlyClassifiedTr/batchSize)*100;
    errorTr(count) = error/batchSize;
    
    plot(count, errorTr(count),'b-*');
    errorV = 0;
    classificationErrorsV = 0;
    correctlyClassifiedV = 0;
    for k2 = 1: size(inputValidation,2)
        inputVector = inputValidation(:,k2);
        targetVector = targetValuesValidation(:,k2);
        output_val=activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector));
        errorV = errorV + norm(output_val - targetVector, 2);
        max = -100;
        classVal(k2) = 1;
        for ii = 1: size(output_val, 1)
            if output_val(ii) > max
                max = output_val(ii);
                classVal(k2) = ii-1;
            end;
        end;
        
        if classVal(k2) == Validationlabels(k2)
            correctlyClassifiedV = correctlyClassifiedV + 1;
        else
            classificationErrorsV = classificationErrorsV + 1;
        end;
        
    end;
    classification_accuracyV(count)=(correctlyClassifiedV/size(inputValidation,2))*100;
    errorVal(count) = errorV/size(inputValidation,2);
    
    plot(count, errorVal(count),'g-o');
    %xlabel('Number of epochs');
    ylabel('MSE');
    title('Learning Curve Of Neural Network with 1 hidden layer(150)');
end;


Testlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
Testinglabels=Testlabels(1:10000,:);

% Transform the labels to correct target values for validation.
targetValuesTesting = 0.*ones(10, size(Testinglabels, 1));
for n = 1: size(Testinglabels, 1)
    targetValuesTesting(Testinglabels(n) + 1, n) = 1;
end;
classificationErrorsT = 0;
correctlyClassifiedT = 0;
confusionmat=zeros(10,10);
for k = 1: size(inputTesting,2)
    inputVector = inputTesting(:,k);
    targetVector = targetValuesTesting(:,k);
    output_val=activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector));
    max = -100;
    classVal(k) = 1;
    for ii = 1: size(output_val, 1)
        if output_val(ii) > max
            max = output_val(ii);
            classVal(k) = ii-1;
        end;
    end;
    
    if classVal(k) == Testinglabels(k)
        correctlyClassifiedT = correctlyClassifiedT + 1;
        confusionmat(classVal(k)+1,classVal(k)+1)=confusionmat(classVal(k)+1,classVal(k)+1)+1;
    else
        classificationErrorsT = classificationErrorsT + 1;
        confusionmat(Testinglabels(k)+1,classVal(k)+1)=confusionmat(Testinglabels(k)+1,classVal(k)+1)+1;
    end;
    
end;
classification_accuracyT=(correctlyClassifiedT/size(inputTesting,2))*100;