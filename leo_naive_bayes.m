
close all; clear all;
clc;
load('pslen-pitstopits-data-512.mat');

aa = data(1).H;
HH = [];
YY = [];
for i = 1:size(data,2)
    XX = data(i).X';
    XX = reshape(XX,1,[]);
    HH = [HH ; data(i).pre data(i).H XX double(data(i).Y)];

end
Data = array2table(HH);
% hypopts = struct('ShowPlots',false,'Verbose',0,'UseParallel',true);
% poolobj = gcp;
% 
% % Ensemble of Decision trees
% %mdl = fitcensemble(Data,'HH112','Learners','tree', ...
%    % 'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);
% 
% mdls{1} = fitcsvm(Data,'HH112','KernelFunction','polynomial','Standardize','on', ...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);
% mdls{2} = fitcsvm(Data,'HH112','KernelFunction','gaussian','Standardize','on', ...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);
% 
% % Decision tree
% mdls{3} = fitctree(Data,'HH112', ...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);
% 
% % Ensemble of Decision trees
% mdls{4} = fitcensemble(Data,'HH112','Learners','tree', ...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);

% Naive Bayes
mdl= fitcnb(Data,'HH112', ...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);



save('ensemblesModel-pslen-pitts2pitts-data-512','mdl');
