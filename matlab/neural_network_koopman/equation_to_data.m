
function data = equation_to_data()
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
    
    data = zeros(1681, 501, 3);
    data(:, :, 1) = t_data;
    data(:, :, 2) = x1_data;
    data(:, :, 3) = x2_data;
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
