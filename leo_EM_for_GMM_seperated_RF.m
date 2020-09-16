
close all; clear all;
clc;
load('/home/leo/mega/pslen-1/models/vd16_tokyoTM_to_paris512_data.mat');


fprintf( 'Creating PSLEN Model \n')

%  load(pslen_config.save_pslen_data,'data');
HH = [];
for i = 1:size(data,2)
XX = data(i).X';
XX = reshape(XX,1,[]);
HH = [HH ; data(i).pre data(i).H XX double(data(i).Y)];
end
Data = array2table(HH);
hypopts = struct('ShowPlots',false,'Verbose',0,'UseParallel',false);

% Decision tree
mdls{1} = fitctree(Data,'HH112', ...
'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);
% Random fitrensemble

mdls{2} = fitrensemble(Data,'HH112');

t = templateTree('MaxNumSplits',1);
mdls{3} = fitrensemble(Data,'HH112','Learners',t,'CrossVal','on');

maxMinLS = 20;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
numPTS = optimizableVariable('numPTS',[1,size(Data,2)-1],'Type','integer');
hyperparametersRF = [minLS; numPTS];

mdls{4} = TreeBagger(300,Data,'HH112','Method','regression',...
'OOBPrediction','On');

mdls{6} = TreeBagger(50,Data,'HH112','Method','regression',...
'OOBPrediction','On');


results = bayesopt(@(params)oobErrRF(params,Data),hyperparametersRF,...
'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);

bestOOBErr = results.MinObjective;
bestHyperparameters = results.XAtMinObjective;

mdls{8} = TreeBagger(300,Data,'HH112','Method','regression',...
'MinLeafSize',bestHyperparameters.minLS,...
'NumPredictorstoSample',bestHyperparameters.numPTS);

save('/home/leo/mega/pslen-1/models/vd16_tokyoTM_to_paris512_mdls.mat','mdls');
fprintf( 'PSLEN Model is created. \n')

    
function oobErr = oobErrRF(params,X)
%oobErrRF Trains random forest and estimates out-of-bag quantile error
%   oobErr trains a random forest of 300 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.
randomForest = TreeBagger(300,X,'HH112','Method','regression',...
'OOBPrediction','on','MinLeafSize',params.minLS,...
'NumPredictorstoSample',params.numPTS);
oobErr = oobQuantileError(randomForest);
end
