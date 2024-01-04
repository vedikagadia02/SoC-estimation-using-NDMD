numLatentChannels = 32;
inputSize = [1 2 NaN];
projectionSize = [1 3 NaN];
inputFormat = "CSB";
indim = 2;
obsdim = 3;

layersE = [
    inputLayer(inputSize,inputFormat)
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(obsdim)
    ];

layersD = [
    inputLayer(projectionSize,inputFormat)
    % featureInputLayer(3)
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(indim)
    ];

netE = dlnetwork(layersE);
netD = dlnetwork(layersD);

numEpochs = 5;
miniBatchSize = 128;
learnRate = 1e-3;

[training_input_data, training_target_data] = dataLoader();

dsTrain = arrayDatastore(training_input_data,IterationDimension=1);
numOutputs = 1;
disp("DS train")
disp(size(dsTrain))
mbq = minibatchqueue(dsTrain,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="BCS", ...
    PartialMiniBatch="discard");
disp("mini batch")
disp(size(mbq))

trailingAvgE = [];
trailingAvgSqE = [];
trailingAvgD = [];
trailingAvgSqD = [];

numObservationsTrain = size(training_input_data,1);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info="Epoch", ...
    XLabel="Iteration");


epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;

        % Read mini-batch of data.
        X = next(mbq);
        disp("in while loop!")
        disp(size(X));
        % Evaluate loss and gradients.
        [loss,gradientsE,gradientsD] = dlfeval(@modelLoss,netE,netD,X);

        % Update learnable parameters.
        [netE,trailingAvgE,trailingAvgSqE] = adamupdate(netE, ...
            gradientsE,trailingAvgE,trailingAvgSqE,iteration,learnRate);

        [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, ...
            gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);

        % Update the training progress monitor. 
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100*iteration/numIterations;
    end
end

function X = preprocessMiniBatch(dataX) 

% Concatenate.
X = cat(1,dataX{:});

end

function [loss,gradientsE,gradientsD] = modelLoss(netE,netD,X)
Z = forward(netE,X);
disp("in modelLoss function!")
disp("size of X!")
disp(size(X))
disp("size of Z!")
disp(size(Z))
% Forward through decoder.
X1 = reshape(Z, 3, 1, 128);
disp("size of x1!")
disp(size(X1))
Xin = dlarray(X1,'SCB');
disp("size of xin!")
disp(size(Xin))
Y = forward(netD,Xin);
disp("size of Y!")
disp(size(Y))
Y1 = reshape(Y, 2, 1, 128);
Yin = dlarray(Y1,'SCB');
% Calculate loss and gradients.
loss = elboLoss(Yin,X);
[gradientsE,gradientsD] = dlgradient(loss,netE.Learnables,netD.Learnables);

end

function loss = elboLoss(Y,T)

% Reconstruction loss.
reconstructionLoss = mse(Y,T);
loss = reconstructionLoss;

end