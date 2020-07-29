clc;
clear all;
% mu = [0 0];
% sigma = [0.25 0.3; 0.3 1];
% x1 = -3:0.2:3;
% x2 = -3:0.2:3;
% [X1,X2] = meshgrid(x1,x2);
% X = [X1(:) X2(:)];
% y = mvnpdf(X,mu,sigma);
% y = reshape(y,length(x2),length(x1));
% surf(x1,x2,y)
% caxis([min(y(:))-0.5*range(y(:)),max(y(:))])
% axis([-3 3 -3 3 0 0.4])
% xlabel('x1')
% ylabel('x2')
% zlabel('Probability Density')
% True curve.
fun = @(x) x.*sin(x);
xx  = linspace(0,10,100)';
yy  = fun(xx);

% Example points (from doc). You can do xd = linspace(1,9,nd)' as well 
% but you will get a different fit.    
xd = [1,3,5,6,7,8]';
yd = fun(xd);

% Fit a GP model. Initialize 'Sigma' to a small value. A GP estimates
% its parameters by maximizing the marginal log likelihood. Depending
% on the data, the marginal log likelihood can have multiple local
% optima corresponding to different interpretations of the data.
% Initializing 'Sigma' to a small value discourages the high noise
% variance interpretation of the data.
gp = fitrgp(xd,yd,'KernelFunction','squaredexponential','sigma',0.1,'verbose',1);

% Plot.
figure(1); clf
plot(xx,yy, 'r-.')
hold on;

[ypred,~,yint] = predict(gp,xx);

plot(xx,ypred, 'g-');
plot(xx,yint(:,1),'k-');
plot(xx,yint(:,2),'m-');

plot(xd,yd,'ro');

legend('f(x) = x.*sin(x)','GPR predictions','Lower 95% interval','Upper 95% interval','Observations','Location','Best');