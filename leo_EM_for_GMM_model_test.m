%https://ch.mathworks.com/help/stats/fitgmdist.html#bt9d3kh-1


close all; clear;
aa = load('pslen-tokyo2tokto-GMM-87.mat');
X = aa.gmm_gt(:,2:3);
label = aa.gmm_gt(:,1)+1;
gm = fitgmdist(X,2,'Start',label);


scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
title('Simulated Data')
gmPDF = @(x,y)reshape(pdf(gm,[x(:) y(:)]),size(x));
hold on
h = fcontour(gmPDF,[-8 6]);
title('Simulated Data and Contour lines of pdf');



save('pslen-tokyo2tokto-GMM-model-87.mat','model');

figure
y = [zeros(15750,1);ones(15750,1)];
h = gscatter(X(:,1),X(:,2),y);
hold on
gmPDF = @(x1,x2)reshape(pdf(GMModel,[x1(:) x2(:)]),size(x1));
g = gca;
fcontour(gmPDF,[g.XLim g.YLim])
title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
legend(h,'Model 0','Model1')
hold off