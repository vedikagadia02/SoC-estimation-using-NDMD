classdef KoopmanNetwork < handle
    properties
        encoder
        decoder
        obsdim
        kMatrixDiag
        kMatrixUT
    end
    
    methods
        function obj = KoopmanNetwork(indim, obsdim)
            obj.obsdim = obsdim;
            
            obj.encoder = [
                fullyConnectedLayer(100)
                reluLayer
                fullyConnectedLayer(100)
                reluLayer
                fullyConnectedLayer(100)
                reluLayer
                fullyConnectedLayer(obsdim)
            ];

            obj.decoder = [
                fullyConnectedLayer(100)
                reluLayer
                fullyConnectedLayer(100)
                reluLayer
                fullyConnectedLayer(100)
                reluLayer
                fullyConnectedLayer(indim)
            ];

            obj.kMatrixDiag = rand(1, obsdim);
            obj.kMatrixUT = 0.01 * randn(1, int32(obsdim*(obsdim-1)/2));

            obj.initializeWeights(obj.encoder);
            obj.initializeWeights(obj.decoder);

            disp(['Total number of parameters: ', num2str(obj.numParameters())]);
        end

        function weights = customWeightInitializer(obj, sz, meanValue, stdDeviation)
            weights = meanValue + stdDeviation * randn(sz);
        end
        
        function initializeWeights(obj, layers)
            for i = 1:numel(layers)
                if isa(layers(i), 'nnet.cnn.layer.FullyConnectedLayer')
                    meanValue = 0;
                    stdDeviation = 2/layers(i).InputSize(2);
                    layers(i).Weights = obj.customWeightInitializer(size(layers(i).Weights), meanValue, stdDeviation);
                    layers(i).Bias = zeros(layers(i).OutputSize, 1);
                end
            end
        end

        function [g, x0] = forward(obj, x)
            g = predict(obj.encoder, x);
            x0 = predict(obj.decoder, g);
        end

        function x0 = recover(obj, g)
            x0 = predict(obj.decoder, g);
        end

        function gnext = koopmanOperation(obj, g, s)
            kMatrix = zeros(obj.obsdim, obj.obsdim);
            utIdx = triu(true(obj.obsdim), 1);
            diagIdx = eye(obj.obsdim);
            kMatrix(utIdx) = obj.kMatrixUT;
            kMatrix(utIdx') = -obj.kMatrixUT;
            kMatrix(diagIdx) = max(0, obj.kMatrixDiag);

            gnext = g * kMatrix;
            for i = 1:s
                if i == 1
                    continue;
                else
                    gnext = gnext * kMatrix;
                end
            end
        end

        function kMatrix = getKoopmanMatrix(obj)
            kMatrix = zeros(obj.obsdim, obj.obsdim);
            utIdx = triu(true(obj.obsdim), 1);
            diagIdx = eye(obj.obsdim);
            kMatrix(utIdx) = obj.kMatrixUT;
            kMatrix(utIdx') = -obj.kMatrixUT;
            kMatrix(diagIdx) = max(0, obj.kMatrixDiag);
            kMatrix = dlarray(kMatrix, 'CU');

        end

        function count = numParameters(obj)
            count = 0;
            layers = [obj.encoder; obj.decoder];
            for i = 1:numel(layers)
                if isa(layers(i), 'nnet.cnn.layer.FullyConnectedLayer')
                    count = count + numel(layers(i).Weights) + numel(layers(i).Bias);
                end
            end
        end

        function backprop(obj, ae, input_t, output_t, )
    end
end
