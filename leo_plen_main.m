clc;
%clear all;

addpath(genpath(pwd));
setup; 
% check input list 'xinput list'
%system('xinput set-prop 17 "Synaptics Two-Finger Scrolling" 1 0');
%system('xinput set-prop 12 "Synaptics Two-Finger Scrolling" 1 0');

  
%   ==>> 3.000000 75.000000 8.000000 82.000000 4.000000 
%24
iTestSample_Start= 1; startfrom = 1;  show_output = 0;

%3 -> 54.000000 72.000000 97.000000 78.000000 16.000000 


%% TOKYO DATASET
%netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white'; % netID= 'caffe_tokyoTM_conv5_vlad_preL2_intra_white';

%dbTest= dbTokyo247();
%datasets_path = 'datasets/Test_247_Tokyo_GSV'; %% PC

%save_path = '/home/leo/mega/vt-6';
%save_path = '/home/leo/mega/vt-7-pitts2tokyo';

%datasets_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV'; %% LAPTOP
%save_path = '/home/leo/MEGA/vt-6';

%datasets_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV'; %% LAPTOP
%save_path = '/home/leo/MEGA/Tokyo24-boxed-vt-6';
%save_path_all = 'pslen-results/pslen-tokyo2tokto-vt-6.mat';

%% PITTSBURGH DATASET
netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';
%1061 tk ho gia hai
dbTest= dbPitts('30k','test');
datasets_path = 'datasets/Test_Pitts30k';
save_path = '/home/leo/mega/Pitts-ori-vt-6';

%% Pitts 2 TOKYO DATASET
netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';

dbTest= dbTokyo247();

datasets_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV'; %% LAPTOP
save_path = '/home/leo/MEGA/vt-7-pitts2tokyo';
save_path_all = '';


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

% To create new output bin files on the datasets
%serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);

%serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1); % adjust batchSize depending on your GPU / network size


[recall, ~, ~, opts]= leo_slen_testFromFn(dbTest, dbFeatFn, qFeatFn, plen_opts);
%[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
save_results = strcat(paths.outPrefix,'plots/pitts30k2tokyo30k','ori.mat');
%save_results = strcat(paths.outPrefix,'plots/pitts30k2pitts30k','ori.mat');
recallNs = opts.recallNs;
save(char(save_results), 'recall','recallNs');

x = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_plot.mat'));
x1 = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_3_plot.mat'));
x2 = load(char('pslen-results/pslen_tokyo2tokyo_vt_6_5_plot.mat')); % nearly equal to netvlad

ori = load(char('/home/leo/docker_ws/datasets/netvlad-original-output/plots/vd16_tokyoTM_conv5_3_vlad_preL2_intra_white_real.mat'));



plot(opts.recallNs, recall, 'bo-', ...
     x2.recallNs, x2.recall, 'go-', ...
     x1.recallNs, x1.recall, 'ro-', ...
     ori.recallNs, ori.recall, 'bo-'...
     ); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none');

