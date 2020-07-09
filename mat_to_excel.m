load('pslen-tokyo2tokto-GMM-87.mat');
filename = 'data-87.xlsx';
writematrix(gmm_gt,filename,'Sheet',1)


save('pslen-tokyo2tokto-GMM-model-trained.mat','trainedModel');

%  yfit = trainedModel.predictFcn(T) 
