function loss_ae = train_ae(training_input_data, indim, obsdim)
    loss_ae = 0;
    learnRate=1e-3;
    disp("ran 1 ")
    ae = KoopmanNetwork(indim, obsdim,true);
    disp(ae.encoder_layer())
    netE = dlnetwork(ae.encoder_layer());
    summary(netE)
    netD = dlnetwork(ae.decoder_layer());
    miniBatchSize = 256;

    dsTrain = arrayDatastore(training_input_data,IterationDimension=1);
    numOutputs = 1;
    mbq = minibatchqueue(dsTrain,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SCB", ...
    PartialMiniBatch="discard");

    iteration = 0;
    while hasdata(mbq)
        iteration = iteration +1;
        input_t = next(mbq);
        disp("ran 4")
        %input_t = training_input_data(idx, :, :);
        %input_t = squeeze(input_t);
        %disp(size(input_t))
        [loss,gradientsE,gradientsD] = dlfeval(@modelLoss,netE,netD,input_t);
        [netE] = adamupdate(netE, ...
            gradientsE, iteration,learnRate);

        [netD] = adamupdate(netD, ...
            gradientsD,iteration,learnRate);
        loss_ae = loss_ae + loss;
        
    end
end

function X = preprocessMiniBatch(dataX)

% Concatenate.
X = cat(1,dataX{:});

end

function [loss,gradientsE,gradientsD] = modelLoss(netE,netD,input_t)
    % Forward through encoder.
    encoder_output_t = forward(netE,input_t);
    % Forward through decoder.
     output_t = forward(netD,encoder_output_t);
    % Calculate loss and gradients.
    loss = immse(output_t, input_t );
    [gradientsE,gradientsD] = dlgradient(loss,netE.Learnables,netD.Learnables);
end