function loss_ae = train_ae(training_input_data, indim, obsdim)
    loss_ae = 0;
    lr=1e-3;
    ae = KoopmanNetwork(indim, obsdim,true)
    net = dlnetwork([ae.encoder,ae.decoder]);
    for idx = 1:750399
        input_t = training_input_data(idx, :, :);
        encoder_output_t, output_t = ae.forward(training_input_data);
        loss_rec = immse(input_t, output_t);
        gradients = dlgradient(loss_rec,network.Learnables,RetainData=true);
        net = adamupdate(net,gradients,lr);
        loss_ae = loss_ae + loss_rec;
        
    end
end