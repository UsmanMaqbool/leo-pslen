function pslen_model(pslen_config) 
    
    fprintf( 'Creating PSLEN Model \n')

    load(pslen_config.save_pslen_data,'data');
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
    
    mdls{4} = TreeBagger(300,Data,'HH112,'Method','regression',...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS);
  
    save(pslen_config.save_pslen_data_mdl,'mdls');
    fprintf( 'PSLEN Model is created. \n')
end