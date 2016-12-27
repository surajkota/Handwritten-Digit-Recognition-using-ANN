figure;
plot(errorTr)
hold on
plot(errorVal)
legend('Training Dataset', 'Validation Dataset');
xlabel('Epochs');
ylabel('MSE');
title('Learning Curve with ReLU activation function 30 epochs, 100 hidden units and 0.01 learning rate');
figure;
plot(classification_accuracyTr)
hold on 
plot(classification_accuracyV)
legend('Training Dataset', 'Validation Dataset');
ylabel('Classification accuracy');
xlabel('Epochs');
title('Classification accuracy for ReLU activation function 30 epochs, 100 hidden units and 0.01 learning rate');