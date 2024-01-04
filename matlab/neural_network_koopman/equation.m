function [data_temp, data] = equation()
    tmax = 50;
    dt = 0.1;
    lamda = -1;
    mu = -0.1;

    x1_data = zeros(1681, 501,1);
    x2_data = zeros(1681, 501,1);
    t_data = zeros(1681, 501,1);

    idx = 1;
    for x1_0 = -2:0.1:2
        for x2_0 = -2:0.1:2
            [t, X] = solve_equation(tmax, dt, x1_0, x2_0, lamda, mu);
            x1 = X(:,1);
            x2 = X(:,2);
            x1_data(idx, :, 1) = x1;
            x2_data(idx, :, 1)= x2;
            t_data(idx, :, 1) = t;
            idx = idx + 1;
        end
    end
    
    data_temp = zeros(1681, 501, 3);
    data_temp(:, :, 1) = t_data;
    data_temp(:, :, 2) = x1_data;
    data_temp(:, :, 3) = x2_data;
    data = zeros(1, 501, 3, 1681);
    for i = 1:1681
        data(:, :, :, i) = data_temp(i, :, :);
    end
    save('equation.mat', 'data');
end

function [t, X] = solve_equation(tmax, dt, x1_0, x2_0, lamda, mu)
    t = 0:dt:tmax;
    X0 = [x1_0, x2_0];
    [t, X] = ode45(@(t, X) deriv(t, X, lamda, mu), t, X0);
end

function dXdt = deriv(t, X, lamda, mu)
    x1 = X(1);
    x2 = X(2);
    dx1dt = -0.1*x1;
    dx2dt = -1*(x2-x1^2);
    dXdt = [dx1dt; dx2dt];
end
