function [netE, netK, netD] = define_nn_arch()

inputSize = [1 2 NaN];
projectionSize = [1 3 NaN];
inputFormat = "CSB";
indim = 2;
obsdim = 3;

layersE = [
    inputLayer(inputSize,inputFormat)
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(obsdim)
    ];

layersK = [
    inputLayer(projectionSize, inputFormat)
    fullyConnectedLayer(obsdim)
    ];

layersD = [
    inputLayer(projectionSize,inputFormat)
    % featureInputLayer(3)
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(indim)
    ];

netE = dlnetwork(layersE);
netK = dlnetwork(layersK);
netD = dlnetwork(layersD);

end