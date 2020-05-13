clc;
%clear all;

addpath(genpath(pwd));
setup; 
% check input list 'xinput list'
%system('xinput set-prop 17 "Synaptics Two-Finger Scrolling" 1 0');
%system('xinput set-prop 12 "Synaptics Two-Finger Scrolling" 1 0');


%==>> 3.000000 75.000000 8.000000 82.000000 4.000000 
%24
iTestSample_Start= 1; startfrom = 1;  show_output = 0;



%% TOKYO DATASET
netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white'; % netID= 'caffe_tokyoTM_conv5_vlad_preL2_intra_white';

dbTest= dbTokyo247();
% datasets_path = 'datasets/Test_247_Tokyo_GSV'; %% PC
%save_path = '/home/leo/mega/vt-6';

datasets_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV'; %% LAPTOP
save_path = '/home/leo/MEGA/Tokyo24-boxed-vt-6';
save_path_all = 'pslen-results/pslen-tokyo2tokto-vt-6.mat';

%% PITTSBURGH DATASET
%netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';
%dbTest= dbPitts('30k','test');
%datasets_path = 'datasets/Test_Pitts30k';
%save_path = 'datasets/Test_Pitts30k/vt-3';

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
            'cropToDim',            0 ...
            );



paths= localPaths();

load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );

%%
net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet

%%

dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);  % just to create the files in the out folder
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);    % just to create the files in the out folder

%To create new output/*bin files on the datasets
%serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1); % adjust batchSize depending on your GPU / network size
%serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);

%Test the features by loading the bin files
[recall, ~, ~, opts]= leo_slen_testFromFn(dbTest, dbFeatFn, qFeatFn, plen_opts);
save_results = strcat('pslen-results/pslen_tokyo2tokyo_','vt_6_2_plot.mat');
recallNs = opts.recallNs;
save(char(save_results), 'recall','recallNs');

x = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_plot.mat'));
x1 = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_1_plot.mat'));
x2 = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_2_plot.mat')); % nearly equal to netvlad

ori = load(char('/home/leo/docker_ws/datasets/netvlad-original-output/plots/vd16_tokyoTM_conv5_3_vlad_preL2_intra_white_real.mat'));



plot(opts.recallNs, recall, 'ro-', ...
     x1.recallNs, x1.recall, 'go-', ...
     x.recallNs, x.recall, 'go-', ...
     ori.recallNs, ori.recall, 'bo-'...
     ); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none');

