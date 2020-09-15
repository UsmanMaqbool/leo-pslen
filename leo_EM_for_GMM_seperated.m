
close all; clear all;
clc;
load('pslen-v12-vd16_pitts2paris-data-512.mat','data');
aa = data(1).H;
HH = [];
YY = [];
for i = 1:size(data,2)
    XX = data(i).X';
    XX = reshape(XX,1,[]);
    HH = [HH ; data(i).pre data(i).H XX double(data(i).Y)];

end
Data = array2table(HH);
hypopts = struct('ShowPlots',false,'Verbose',0,'UseParallel',false);
poolobj = gcp;

% Decision tree
%mdls{1} = fitctree(Data,'HH112', 'OptimizeHyperparameters','auto');
mdls{1} = fitctree(Data,'HH112', ...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);


mdls{2} = fitrensemble(Data,'HH112');

% Naive Bayes

% mdls{3} = fitcnb(Data,'HH112', ...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);


save('pslen-v12-vd16_pitts2paris-data-512-mdls.mat','mdls');
fprintf( 'Done :)')
