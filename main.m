clc;
clear all;

addpath(genpath(pwd));
setup; 

paths= localPaths();
pslen_config = pslen_settings(paths);

netID = pslen_config.netID;
load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );


%% Check PSLEN model available

if ~exist(pslen_config.save_pslen_data_mdl, 'file') && strcmp(pslen_config.pslen_on,'paris')
    pslen_config.createPslenModel = true;
    dbTest= dbVGG('paris');
    pslen_config.datasets_path = paths.dsetRootParis; %% PC
    pslen_config.query_folder = 'images';    
    
    qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);   % just to create the files in the out folder
    dbFeatFn = sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);  % just to create the files in the out folder

    % Create models if not available
    if ~exist(qFeatFn, 'file')
        %%
        net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet
        serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
    end

    if ~exist(qFeatFn, 'file')
        serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1); % adjust batchSize depending on your GPU / network size
    end
    
    [~,~,~,~,~] = pslen_testFromFn(dbTest, dbFeatFn, qFeatFn, pslen_config, [], 'cropToDim', pslen_config.cropToDim);

end

%% Whole Process

qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);   % just to create the files in the out folder
dbFeatFn = sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);  % just to create the files in the out folder

dbTest = pslen_config.dbTest;

% Create models if not available
if ~exist(qFeatFn, 'file')
    %%
    net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet
    serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
end

if ~exist(qFeatFn, 'file')
    serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1); % adjust batchSize depending on your GPU / network size
end

% Use PSLEN model
[~, ~,recall,recall_ori, opts]= pslen_testFromFn(dbTest, dbFeatFn, qFeatFn, pslen_config, [], 'cropToDim', pslen_config.cropToDim);


%% Results
pslen_results = [opts.recallNs',recall_ori*100];
netvlad_results = [opts.recallNs',ori.recall*100];


dlmwrite(pslen_config.pslen_results_fname,PSLEN_results,'delimiter',' ');
dlmwrite(pslen_config.netvlad_results_fname,netvlad_results,'delimiter',' ');


%save(char(save_results), 'recall','recallNs', 'recall_ori');
pre = load(save_results);

plot(opts.recallNs, pre.recall, 'bo-', ...
     opts.recallNs, recall_ori, 'ro-' ,...
     opts.recallNs, recall, 'go-' ...
     ); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none'); legend({'Previous Best','Original', 'New'});


