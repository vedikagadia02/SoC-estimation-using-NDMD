function loss_ae = train_ae(training_input_data, indim, obsdim)
    loss_ae = 0;
    lr=1e-3;
    disp("ran 1 ")
    ae = KoopmanNetwork(indim, obsdim,true);
    disp("ran 2 ")
    netE = dlnetwork(ae.encoder_layer());
    disp("ran 3 ")
    netD = dlnetwork(ae.decoder_layer());
    for idx = 1:7
        disp("ran 4")
        input_t = training_input_data(idx, :, :);
        input_t = squeeze(input_t);
        disp(size(input_t))
        [encoder_output_t, output_t] = ae.forward(input_t);
        loss_rec = immse(input_t, output_t);
        [gradientsE, gradientsD] = dlgradient(loss_rec,netE.Learnables,netD.Learnables);
        netE = adamupdate(netE,gradientsE,lr);
        netD = adamupdate(netD,gradientsD,lr);
        
        loss_ae = loss_ae + loss_rec;
        
    end
end