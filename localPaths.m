function paths= localPaths()
    
    % --- dependencies
    
    % refer to README.md for the information on dependencies
    paths.libReljaMatlab= 'depends/relja_matlab/';
    paths.libMatConvNet= '3rd-party-support/matconvnet/'; % should contain matlab/
    
    % If you have installed yael_matlab (**highly recommended for speed**),
    % provide the path below. Otherwise, provide the path as 'yael_dummy/':
    % this folder contains my substitutes for the used yael functions,
    % which are **much slower**, and only included for demonstration purposes
    % so do consider installing yael_matlab, or make your own faster
    % version (especially of the yael_nn function)
    paths.libYaelMatlab= 'yael_dummy/';
    
        % --- dataset specifications

    paths.dsetSpecDir= 'datasets/datasets-specs';
    %paths.dsetSpecDir= '/home/leo/docker_ws/datasets/datasets-specs';

    % --- dataset locations
   % paths.dsetRootPitts= '/home/leo/docker_ws/datasets/Pittsburgh-all/Pittsburgh/'; % should contain images/ and queries/
    
    paths.dsetRootPitts= '/mnt/0287D1936157598A/docker_ws/datasets/NetvLad/view-tags/Pittsburgh_Viewtag_3_rgb/';
    
    % CLuster
    %paths.dsetRootTokyo247= '/cluster/scratch/mbhutta/Test_247_Tokyo_GSV/'; % should contain images/ and query/
    % XPS
    paths.dsetRootTokyo247= 'datasets/Test_247_Tokyo_GSV/'; % should contain images/ and query/
    %paths.dsetRootTokyo247= '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/'; % should contain images/ and query/
    %paths.dsetRootTokyoTM= '/home/leo/docker_ws/datasets/tokyoTimeMachine/'; % should contain images/
    paths.dsetRootTokyoTM= '/home/leo/docker_ws/datasets/tinyTimeMachine/'; % should contain images/
    paths.dsetRootOxford= 'datasets/test_oxford/'; % should contain images/ and groundtruth/, and be writable

    paths.dsetRootParis= '/mnt/0287D1936157598A/docker_ws/docker_ws/Maqbool/datasets/test_paris/'; % should contain images/ (with subfolders defense, eiffel, etc), groundtruth/ and corrupt.txt, and be writable
    paths.dsetRootHolidays= '/mnt/0287D1936157598A/docker_ws/docker_ws/Maqbool/datasets/test-inria/'; % should contain jpg/ for the original holidays, or jpg_rotated/ for rotated Holidays, and be writable
    
    
    % models used in our paper, download them from our research page
    % paths.ourCNNs= '~/Data/models/';
   % paths.ourCNNs= '/mnt/0287D1936157598A/docker_ws/datasets/NetvLad/models_v103_pre-trained/';
   paths.ourCNNs= 'datasets/models_v103_pre-trained/';
   %paths.ourCNNs= '/home/leo/docker_ws/datasets/models_v103_pre-trained/';

    % --- pretrained networks
    % off-the-shelf networks trained on other tasks, available from the MatConvNet

    %% HK PC
    %% paths.outPrefix= '/mnt/0287D1936157598A/docker_ws/datasets/netvlad-original-output/';
    %% LAPTOP
    paths.outPrefix= 'datasets/netvlad-original-output/';
  %  paths.outPrefix= '/home/leo/docker_ws/datasets/netvlad-original-output/';
end
