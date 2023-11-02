clear all, close all
dt=0.01; T=50; t = 0:dt:T;
mu = -0.05; lamda = 1;

dynamic_system = @(t,x)([mu*x(1); ...
    lamda*(x(2)^2-x(1));]);
ode_options = odeset('RelTol',1e-10,'AbsTol',1e-11);

input =[];output=[];
for j=1:100
    x0=[-5+0.01*j,5-0.01*j];
    [t,y] = ode45(dynamic_system,t,x0,ode_options);
    input = [input;y(1:end-1,:)];
    output = [output;y(2:end,:)];
    plot(y(:,1),y(:,2)), hold on 
    plot(x0(1),x0(2),'ro')
end
grid on,view(-20,20)

net = feedforwardnet ([10 10 10]);
net.layers{1}.transferFcn='logsig';
net.layers{2}.transferFcn='radbas';
net.layers{3}.transferFcn='purelin';

net=train(net,input.',output.');

figure(2)
x0 = [-5, 5].';
[t,y] = ode45(dynamic_system,t,x0);
plot(y(:,1),y(:,2)), hold on 
plot(x0(1),x0(2),'ro','LineWidth',2)
grid on 

ynn(1,:) = x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.';x0=y0;
end
plot(ynn(:,1),ynn(:,2),':','LineWidth',2)


