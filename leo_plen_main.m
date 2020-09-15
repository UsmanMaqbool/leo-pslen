clc;
clear all;

addpath(genpath(pwd));
setup; 


%%1720
iTestSample_Start= 1720; startfrom = 1;  show_output = 0; 
mode = 'test' ; %'training' , 'test'
proj = 'pslen'; %'vt-rgb', 'pslen'
f_dimension = 512;
job_net = 'vd16_pitts30k'; % 'vd16_tokyoTM';   % 'vd16_pitts30k' 
job_datasets = 'pitts30k';  %'tokyo247' 'pitts30k' 'oxford' , 'paris', 'paris-vt-rgb', 'pitts30k-vt-rgb

%%
if strcmp(job_net,'vd16_pitts30k')
    % PITTSBURGH DATASET
    netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';
            %query_folder = 'query';


elseif strcmp(job_net,'vd16_tokyoTM')
    % TOKYO DATASET
    netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
    query_folder = 'query';
    
end

if strcmp(job_datasets,'pitts30k')
    dbTest= dbPitts('30k','test');
    datasets_path = 'datasets/Test_Pitts30k';
        query_folder = 'queries';

elseif strcmp(job_datasets,'pitts30k-vt-rgb')
    dbTest= dbPitts('30k','test');
    datasets_path = '/mnt/0287D1936157598A/docker_ws/datasets/NetvLad/view-tags/Pittsburgh_Viewtag_3_rgb'; %% PC
         query_folder = 'queries';

elseif strcmp(job_datasets,'tokyo247')
    dbTest= dbTokyo247();
    datasets_path = 'datasets/Test_247_Tokyo_GSV'; %% PC
 
elseif strcmp(job_datasets,'oxford')
    dbTest= dbVGG('ox5k');
    datasets_path = 'datasets/test_oxford'; %% PC
    query_folder = 'images';
elseif strcmp(job_datasets,'paris')
    dbTest= dbVGG('paris');
    datasets_path = 'datasets/test_paris'; %% PC
    query_folder = 'images';
elseif strcmp(job_datasets,'paris-vt-rgb')
    dbTest= dbVGG('paris');
    datasets_path = 'datasets/test_paris-vt/3_rbg'; %% PC
    query_folder = 'images';
end

%save_path = strcat('/home/leo/mega/pslen/',job_net,'_to_',job_datasets,'_box_51_plus');
save_path = strcat('/home/leo/mega/pslen/',job_net,'_to_',job_datasets,'_',int2str(f_dimension),'_',proj,'-0.3');

save_results = strcat('plots/',job_net,'_to_',job_datasets,'_pslen_netvlad_results_',int2str(f_dimension),'.mat');
save_path_all = strcat('/home/leo/mega/pslen/all/',job_net,'_to_',job_datasets,'_box_50_plus','.mat');


%% TOKYO DATASET
%netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white'; % netID= 'caffe_tokyoTM_conv5_vlad_preL2_intra_white';

%dbTest= dbTokyo247();fra
%datasets_path = 'datasets/Test_247_Tokyo_GSV'; %% PC

%save_path = '/home/leo/mega/vt-6';
%save_path = '/home/leo/mega/vt-7-pitts2tokyo';

%datasets_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV'; %% LAPTOP
%save_path = '/home/leo/MEGA/vt-6';

%datasets_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV'; %% LAPTOP
%save_path = '/home/leo/MEGA/Tokyo24-boxed-vt-6';
%save_path_all = 'pslen-results/pslen-tokyo2tokto-vt-6.mat';



%% Pitts 2 TOKYO DATASET
%netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';

%dbTest= dbTokyo247();

%datasets_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV'; %% LAPTOP
%datasets_path = 'datasets/Test_247_Tokyo_GSV'; %% LAPTOP

%save_path = '/home/leo/MEGA/vt-7-pitts2tokyo';


%%
plen_opts= struct(...
            'netID',                netID, ...
            'dataset_path',         datasets_path, ...
            'save_path',            save_path, ...
            'save_path_all',        save_path_all, ...
            'vt_type',              3, ...
            'iTestSample_Start',    iTestSample_Start, ...
            'startfrom',            startfrom, ...
            'show_output',          show_output, ...
            'query_folder',         query_folder, ...
            'cropToDim',            f_dimension ...
            );



paths= localPaths();

load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );

%%
net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet

%%

dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name)  % just to create the files in the out folder
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name)   % just to create the files in the out folder

% To create new output bin files on the datasets
% % % % % for query
if strcmp(mode,'training')
serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
% % % % % % for database images
serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1); % adjust batchSize depending on your GPU / network size
% % %  
end
[~, ~,recall,recall_ori, opts]= leo_slen_testFromFn(dbTest, dbFeatFn, qFeatFn, plen_opts, [], 'cropToDim', f_dimension);

recallNs = opts.recallNs;
%save(char(save_results), 'recall','recallNs', 'recall_ori');

ori = load(save_results);



plot(opts.recallNs, ori.recall, 'bo-', ...
     opts.recallNs, recall_ori, 'ro-' ,...
     opts.recallNs, recall, 'go-' ...
     ); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none'); legend({'Previous Best','Original', 'New'});


%save_results = strcat(paths.outPrefix,'plots/pitts30k2tokyo30k','ori.mat');
%save_results = strcat(paths.outPrefix,'plots/pitts30k2pitts30k','ori.mat');
%x = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_plot.mat'));
%x1 = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_3_plot.mat'));
%x2 = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_5_plot.mat')); % nearly equal to netvlad

