%This function is robust and efficient yet the code structure is organized so that it is easy to read. Please try following code for a demo:
close all; clear;
d = 2;
k = 2;
n = 500;
[X,label] = mixGaussRnd(d,k,n);
plotClass(X,label);

m = floor(n/2);
X1 = X(:,1:m);
X2 = X(:,(m+1):end);
% train
[z1,model,llh] = mixGaussEm(X1,k);
figure;
plot(llh);
figure;
plotClass(X1,z1);
% predict
z2 = mixGaussPred(X2,model);
figure;
plotClass(X2,z2);

aa = load('pslen-tokyo2tokto-GMM.mat');
X = aa.gmm_gt(:,2:3)';
label = aa.gmm_gt(:,1)'+1;
[z,model,llh] = mixGaussEm(X,k);
save('pslen-tokyo2tokto-GMM-model.mat','z','model');
