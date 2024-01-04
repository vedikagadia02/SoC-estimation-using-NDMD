function [training_input_data, training_target_data] = dataLoader()
    data = equation_to_data();
    idx = 1:1681;
    endIdx = 500;
    idx(randperm(length(idx)));

    ntrain = 1600;
    training_data = zeros(ntrain, 501, 2);
    training_data(:,:,1) = data(idx(1:ntrain),:,2);
    training_data(:,:,2) = data(idx(1:ntrain),:,3);
    disp(size(training_data))

    training_input_data = [];
    for i = 1:endIdx-31
        training_input_data = [training_input_data; training_data(:, i, :)];
    end
    disp("hi")
    disp(size(training_input_data))
    training_target_data = [];
    for i = 2:endIdx-30
        training_target_data = [training_target_data; training_data(:, i:i+29, :)];
    end
    disp("hi2")
    disp(size(training_target_data))
end