function main()
    ntrain = 1600;
    indim = 2;
    obsdim = 3;
    dt = 0.1;
    s = 30;
    a1 = 1.0;
    a2 = 50.0;
    a3 = 10.0;
    a4 = 1e-6;
    encoder_lr = 0.0005;
    kMatrix_lr = 0.001;
    weight_decay = 1e-7;
    gamma = 0.995;
    epochs = 250;
    epochs_encoder = 5;

    [training_input_data, training_target_data] = dataLoader();
    for idx_ae = 1:5
        loss_ae = train_ae(training_input_data, indim, obsdim);
        disp(loss_ae)
    end
    for idx_model = 1:20
        [loss_model, koopman] = train_model(training_input_data, training_target_data, indim, obsdim);
        disp(loss_model)
        koopman_transformed = log(koopman) / dt;
        for i = 1:size(koopman_transformed, 1)
            formatted_row = arrayfun(@(element) sprintf('%.5f', element), koopman(i, :), 'UniformOutput', false);
            disp(strjoin(formatted_row, ' '));
        end
    end
end