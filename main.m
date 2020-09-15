clc;
clear all;

addpath(genpath(pwd));
setup; 

paths= localPaths();
pslen_config = pslen_settings(paths);

netID = pslen_config.netID;
dbTest = pslen_config.dbTest;


load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
%%
net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet

%%

qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);   % just to create the files in the out folder
dbFeatFn = sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);  % just to create the files in the out folder

% Create models if not available
if ~exist(qFeatFn, 'file')
    serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
end
if ~exist(qFeatFn, 'file')
    serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1); % adjust batchSize depending on your GPU / network size
end


[~, ~,recall,recall_ori, opts]= pslen_testFromFn(dbTest, dbFeatFn, qFeatFn, pslen_config, [], 'cropToDim', pslen_config.f_dimension);

recallNs = opts.recallNs;
%save(char(save_results), 'recall','recallNs', 'recall_ori');

ori = load(save_results);



plot(opts.recallNs, ori.recall, 'bo-', ...
     opts.recallNs, recall_ori, 'ro-' ,...
     opts.recallNs, recall, 'go-' ...
     ); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none'); legend({'Previous Best','Original', 'New'});

pslen_results = [opts.recallNs',recall_ori*100];
netvlad_results = [opts.recallNs',ori.recall*100];

pslen_results_fname = strcat('data/',job_net,'_to_',job_datasets,'_pslen_',int2str(f_dimension),'.dat');
netvlad_results_fname = strcat('data/',job_net,'_to_',job_datasets,'_netvlad_',int2str(f_dimension),'.dat');


dlmwrite(pslen_results_fname,PSLEN_results,'delimiter',' ');
dlmwrite(netvlad_results_fname,netvlad_results,'delimiter',' ');


%save_results = strcat(paths.outPrefix,'plots/pitts30k2tokyo30k','ori.mat');
%save_results = strcat(paths.outPrefix,'plots/pitts30k2pitts30k','ori.mat');
%x = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_plot.mat'));
%x1 = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_3_plot.mat'));
%x2 = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_5_plot.mat')); % nearly equal to netvlad

