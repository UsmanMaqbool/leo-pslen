 function [res, recalls, recalls_ori]= leo_recallAtN(searcher, nQueries, isPos, ns, printN, nSample,db,plen_opts)
    if nargin<6, nSample= inf; end
    
    rngState= rng;
    
    if nSample < nQueries
        rng(43);
        toTest= randsample(nQueries, nSample);
    else
        toTest= 1:nQueries;
    end
    
    assert(issorted(ns));
    nTop= max(ns);
    
    recalls= zeros(length(toTest), length(ns));
    recalls_ori= zeros(length(toTest), length(ns));
    printRecalls= zeros(length(toTest),1);
    
    evalProg= tic;
    %dataset_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV';
    %save_path = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/vt-2';
   
    % detect blackish images  ( plot the boxes)
    % rearrange + use previous knowledge
    

    
    vt_type = plen_opts.vt_type;
    iTestSample_Start=plen_opts.iTestSample_Start; startfrom =plen_opts.startfrom; show_output = plen_opts.show_output;  %test the boxes
    dataset_path = plen_opts.dataset_path; 
    save_path = plen_opts.save_path; 
    %% LEO START
    
    netID= plen_opts.netID;
    % netID= 'caffe_tokyoTM_conv5_vlad_preL2_intra_white';


    paths = localPaths();

    load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );

    %%
    net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet

    if isfile(plen_opts.save_path_all)
     
        load(plen_opts.save_path_all);
    else
        fprintf('pslen-all single file not exits, system will process single images and make pslen-all in the end \n');

        
    end
    
    %addpath(genpath('/mnt/02/docker_ws/docker_ws/netvlad/slen-0.2-box'));
    
    %% EDGE BOX
    %load pre-trained edge detection model and set opts (see edgesDemo.m)

    model=load('edges/models/forest/modelBsds'); model=model.model;
    
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
    % set up opts for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = .85;     % step size of sliding window search0.65
    opts.beta  = .8;     % nms threshold for object proposals0.75
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 200;  % max number of boxes to detect 1e4
    gt=[111	98	25	101];
    opts.minBoxArea = 0.5*gt(3)*gt(4);
    opts.maxAspectRatio = 1.0*max(gt(3)/gt(4),gt(4)./gt(3));
    
    gmm_m =  load('pslen-tokyo2tokto-GMM-model.mat');
    z_gmm = gmm_m.z;
    model_gmm = gmm_m.model;
    
    num_box = 49; % Total = 10 (first one is the full images feature / box)
    
    
    Top_boxes = 10; % will be used.

    top_100 = [];
    total_top = 100; %100;0
    inegatif_i = [];
       gmm_gt = [];

    load('pslen-tokyo2tokto-GMM-model-trained.mat');

 

    for iTestSample= iTestSample_Start:length(toTest)
        
        %Display
        relja_progress(iTestSample, ...
                       length(toTest), ...
                       sprintf('%.4f', mean(printRecalls(1:(iTestSample-1)))), evalProg);
        
     
        iTest= toTest(iTestSample);
        
        [ids ds_pre]= searcher(iTest, nTop); % Main function to find top 100 candidaes
        ds_pre_max = max(ds_pre); ds_pre_min = min(ds_pre);
               ds_pre_mean = mean(ds_pre); ds_pre_var = var(ds_pre);
        %y = normpdf(ds);
        % plot(ds,y)
      % ds_pre_check = 1/sqrt(2*pi*ds_pre_var*ds_pre_var)*exp(-1.((ds-ds_pre_mean)/(2*ds_pre_var*ds_pre_var);
        
        ds = ds_pre - min(ds_pre(:));
        ds = ds ./ max(ds(:)); 

        thisRecall_ori= cumsum( isPos(iTest, ids) ) > 0; % yahan se get karta hai %db.cp (close position)
        ds_pre_gt = single(isPos(iTest, ids));
%         if nnz(thisRecall_ori) == 0
%             ds_pre_gt(iTestSample,1) = 0;
%         else
%             ds_pre_gt(iTestSample,1) = find(thisRecall_ori,1);
% 
%         end
        %% Leo START
                
        qimg_path = strcat(dataset_path,'/',plen_opts.query_folder, '/', db.qImageFns{iTestSample, 1});  
        q_img = strcat(save_path,'/', db.qImageFns{iTestSample, 1});  
        q_feat = strrep(q_img,'.jpg','.mat');
        
        if show_output == 1
        subplot(2,2,1); imshow(imread(char(qimg_path))); %q_img
        db_img = strcat(dataset_path,'/images/', db.dbImageFns{ids(1,1),1});  

        subplot(2,2,2); imshow(imread(char(db_img))); %
        hold;
        end
        
        
        if ~exist(plen_opts.save_path_all, 'file')
                            %         
                            %         if exist(q_feat, 'file')
                            %              x_q_feat = load(q_feat);
                            %              
                            %         else
                            %             im= vl_imreadjpeg({char(qimg_path)},'numThreads', 12); 
                            % 
                            %             I = uint8(im{1,1});
                            %             [bbox, ~] =edgeBoxes(I,model);
                            %             
                            %            % [bbox,im, E, hyt, wyd] = img_Bbox(qimg_path,model);
                            %             
                            %             [hyt, wyd] = size(im{1,1});
            
            if exist(q_feat, 'file')
                 x_q_feat = load(q_feat);
                 x_q_feat_all(iTestSample) = struct ('x_q_feat', x_q_feat); 
            else
                im= vl_imreadjpeg({char(qimg_path)},'numThreads', 12); 

                I = uint8(im{1,1});
                [bbox, ~] =edgeBoxes(I,model);

               % [bbox,im, E, hyt, wyd] = img_Bbox(qimg_path,model);

                [hyt, wyd,~] = size(im{1,1});

                mat_boxes = leo_slen_increase_boxes(bbox,hyt,wyd);

                im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
                query_full_feat= leo_computeRepresentation(net, im, mat_boxes,num_box); % add `'useGPU', false` if you want to use the CPU

                db_bbox_top = [1 1 hyt wyd 1];
                q_bbox = [db_bbox_top ; double(mat_boxes)*16];
                q_bbox = q_bbox (1:num_box+1,:);


                k = Top_boxes;


                % Top 100 sample

                for jj = 1:total_top

                        ds_all_full = [];
                        ids_all = [];

                        db_img = strcat(dataset_path,'/images/', db.dbImageFns{ids(jj,1),1});  
                        im= vl_imreadjpeg({char(db_img)},'numThreads', 12); 
                        I = uint8(im{1,1});
                        [bbox, ~] =edgeBoxes(I,model); % ~ -> Edge (not required)
                        [hyt, wyd,~] = size(im{1,1});   % update the size accordign to the DB images. as images have different sizes. 
                      %  [bbox,im, E, hyt, wyd] = img_Bbox(db_img,model);

                        mat_boxes = leo_slen_increase_boxes(bbox,hyt,wyd);

                        im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
                        feats= leo_computeRepresentation(net, im, mat_boxes,num_box); % add `'useGPU', false` if you want to use the CPU
                        db_bbox_top = [1 1 hyt wyd 1];
                        db_bbox = [db_bbox_top ; double(mat_boxes)*16];
                        db_bbox = db_bbox (1:num_box+1,:);

                        db_bbox_file(jj) = struct ('bboxdb', db_bbox); 

                        fprintf( '==>> %i ~ %i/%i ',iTestSample,jj,total_top );


                        for j = 1:Top_boxes
                            q1 = single(feats(:,j));  %take column of each box
                            [ids1, ds1, top1]= leo_yael_nn(query_full_feat, q1, k);

                            ids1 = [1 ; ids1];          %Take to element
                            ds1 = [top1 ; ds1];         % top element feature

                            ids_all = [ids_all ids1];
                            ds_all_full = [ds_all_full ds1];
                        end

                        clear feats;

                        ids_all_file(jj) = struct ('ids_all', ids_all); 
                        ds_all_file(jj) = struct ('ds_all_full', ds_all_full); 


                end

                  % save the files
             if vt_type == 3
                check_folder = fileparts(q_feat);
                if ~exist(check_folder, 'dir')
                    mkdir(check_folder)
                end
                 save(q_feat,'ds_all_file', 'ids_all_file', 'q_bbox', 'db_bbox_file');
                 clear ids_all_file; clear ds_all_file;
                 x_q_feat = load(q_feat);


            end

            end
        else
            %     load(plen_opts.save_path_all); above loading in
            %     mentioned

            x_q_feat = x_q_feat_all(iTestSample).x_q_feat;
        end
        
        %%% Loading End
%%        
        SLEN_top = zeros(total_top,2); 
       
        exp_ds_pre = exp(-1.*ds_pre);
        ds_pre_diff = diff(ds_pre);
        ds_pre_diff = [ds_pre_diff; 0];
        exp_ds_pre_sum = sum(exp_ds_pre);
        prob_q_db = exp_ds_pre/exp_ds_pre_sum;
        x_q_feat_ids_all = [];
        min_ds_pre_all = [];

        % figure;
            
        for i=startfrom:total_top 
            x_q_feat_ds= x_q_feat.ds_all_file(i).ds_all_full; 
            x_q_feat_ids_all = [x_q_feat_ids_all ;x_q_feat_ds];

        end
        ds_box_all_sum = sum(x_q_feat_ids_all(:));
        
        Prod_ds_box = exp_ds_pre/ds_box_all_sum;
        
        


        for i=startfrom:total_top 
 
            
           %Single File Load
           x_q_feat_ds_all = x_q_feat.ds_all_file(i).ds_all_full;
           x_q_feat_box_q =  x_q_feat.q_bbox;
           x_q_feat_box_db = x_q_feat.db_bbox_file(i).bboxdb;
           x_q_feat_ids_all = x_q_feat.ids_all_file(i).ids_all ; 
           
           
           % Full File Load
           
           x_q_feat.ds_all_file(1).ds_all_full  ;
           
           x_q_feat_ds_all_exp = exp(-1.*x_q_feat_ds_all);
           
           sum_ds_all_Prob = sum(x_q_feat_ds_all_exp(:));
           
           
           ds_all = x_q_feat_ds_all(2:Top_boxes+1,:);
           imgg_mat_box_q =  x_q_feat_box_q(2:num_box+1,:);
           imgg_mat_box_db = x_q_feat_box_db(2:num_box+1,:);
            
           
           x_q_feat_ds_all_1 = x_q_feat_ds_all(:,1);
           x_q_feat_ds_all_1 = x_q_feat_ds_all_1-x_q_feat_ds_all_1(1,1);
            % original dis: 1.25 ds_pre
            db_img = strcat(dataset_path,'/images/', db.dbImageFns{ids(i,1),1});  
            
            ds_pre_inv = 1/ ds_pre(i,1);
            
            ds_all_inv = 1./(ds_all);
            
            diff_ds_all = zeros(Top_boxes,Top_boxes);
            diff_ds_all(1:Top_boxes-1,:) = diff(ds_all);

            diff2_ds_all = diff(diff(ds_all));
            diff2_ds_all_less = diff2_ds_all;
            diff2_ds_all_less(diff2_ds_all_less>0) = 0;
        
            
            ds_all_sub = ds_all(1:Top_boxes,1:Top_boxes);
            
            
            ds_all_less = ds_all_sub-ds_pre(100,1);
            ds_all_less_mean = mean(ds_all_less(:));
             if i > 1
                   relative_diff =  ds_pre(i,1) - ds_pre(i-1,1);
            else
                   relative_diff =  ds_pre(i+1,1) - ds_pre(i,1);
            end
           % exp_relative_diff = exp(-1.*relative_diff); %*exp_related_Box_dis;
            exp_relative_diff = exp(-1.*ds_pre_diff(i,1)); %*exp_related_Box_dis;
  
            s=sign(ds_all_less); s_inv=sign(ds_all_less);
            
            ipositif=sum(s(:)==1);
            inegatif=sum(s(:)==-1);
            inegatif_i=[inegatif_i ;inegatif];

            S_less = s; S_less(S_less>0) = 0; 
            S_less = abs(S_less).*ds_all_less; 
            
            S_less_diff = diff(S_less);
            S_less_mean = sum(sum(S_less/inegatif)); 
           
            D_diff = ds_pre(i,1); %-s_delta_all;
            ds_pre_diff_mean = mean(ds_pre_diff(:));
            
            if i > 1
                   relative_diff =  ds_pre(i,1) - ds_pre(i-1,1);
            else
                   relative_diff =  ds_pre(i+1,1) - ds_pre(i,1);
            end
           % exp_relative_diff = exp(-1.*relative_diff); %*exp_related_Box_dis;
            exp_relative_diff = exp(-1.*ds_pre_diff(i,1)); %*exp_related_Box_dis;
  
            
            
            
            sol_1 = sum(S_less_diff(:));

            S1 = S_less; S1_mean = mean(S1(:),'omitnan');
            S1(S1>S1_mean) = NaN;
            S2 = S1; S2_mean = mean(S2(:),'omitnan');
            S2(S2>S2_mean) = NaN;
            S3 = S2; S3_mean = mean(S3(:),'omitnan');

            S3(S3>S3_mean) = NaN;
            S1(isnan(S1)) = 0;
            S2(isnan(S2)) = 0;
            S3(isnan(S3)) = 0;
            
            D3 = sum(S3(:));
            
             if show_output == 2

                subplot(2,2,1); imshow(imread(char(qimg_path))); %q_img
                subplot(2,2,2); imshow(imread(char(db_img))); %

                
                subplot(2,2,3); h = heatmap(S1);
                subplot(2,2,4); h = heatmap(S3); % with plus is wokring
              %  subplot(2,2,1); h = heatmap(diff_ds_all/ds_pre_inv);
              %  subplot(2,2,2); h = heatmap(diff2_ds_all/ds_pre_inv);
%                fprintf( '==>> Distance %f ~ Greator Values %f %f \n Less Values %f %f ~ Min %f \n',ds_pre(i,1), s_delta_all,ipositif, S_great, inegatif, S_less);
              %  fprintf('%f %f %f %f %f %f %f %f %f %f\n',(Top_boxes*Top_boxes)/inegatif, D_diff, D_diff+ds_all_less_mean, D_diff-(ds_all_less_mean/D), D_diff-s_delta_mat, D_diff+s_delta_mat,ds_all_less_mean+s_delta_mat, s_delta_mat/ds_all_less_mean, D/(D_diff-ds_all_less_mean), D);
               % fprintf('\n For %f %f %f %f %f %f %f %f %f', D_diff, D, boxes_per_less, Top_boxes, inegatif, s_delta_mat, ds_all_less_mean, mean(S_less_diff(:)),min(S_less_diff(:)));
             %   fprintf('%f -> %f %f %f %f %f %f %f %f \n',D_diff,D_diff+ S_less_mean,D_diff+ S_less_n_mean, D_diff+S_less_inv_mean,D_diff+ S_less_inv_n_mean,D_diff+ S_less_diff,D_diff+ S_less_n_diff, D_diff+S_less_inv_diff,D_diff+ S_less_inv_n_diff);

            end
            
            
            S1_diff = diff(S1);
            S2_diff = diff(S2);
            S3_diff = diff(S3);
            S5 = S3(1:Top_boxes-1,:).*S1_diff;
               
               
           S1(isnan(S1)) = 0;
           S7 = S3(1:Top_boxes-2,:).*diff2_ds_all_less;
           S8 = S7.*ds_pre(i,1); %S_less(1:Top_boxes-1,:);
           S6 = S5.*ds_pre(i,1); %S_less(1:Top_boxes-1,:);
           sol_2 = sum(S1(:));
           sol_3= sum(S2(:));
           sol_4 = sum(S3(:));
           sol_5 = sum(S5(:)); %s_delta_mat(:);
           sol_6 = sum(S8(:)); 


           Var_S5 = var(S3,1);
           num_var_s5 = nnz(Var_S5);
           sum_var_s5 = sum(Var_S5);
           mum_var_s5 = num_var_s5*sum_var_s5;
               
              
               
          heat3 = diff2_ds_all_less*ds_pre_inv;
          %check_heat = sum(S8(2,:,:));
          check_heat = 0;
          %D_diff = ds_pre(i,1)-; %-s_delta_all;
           for jj = 1:Top_boxes
               S8_col = S8(:,jj);
               check_heat_mean = mean(S8_col);

                S8_col(S8_col<check_heat_mean) = 0;

                hm = nnz(S8_col);
                if hm >= 2 
                    check_heat = check_heat+ sum(S8_col);
                end
           end

             for jj = 1:Top_boxes

                   S3_nnz = nnz(S3(:,jj));
                   if S3_nnz < 2
                       sum_diff2_ds_all(jj) = 0;

                   end


             end
               nnz_black_check = nnz(sum_diff2_ds_all);
               
               top_candidates = sum_diff2_ds_all;
               
               check_heat = 0;
%              D_diff = ds_pre(i,1)-; %-s_delta_all;
               for jj = 1:Top_boxes
                   S8_col = S8(:,jj);
                    
                    hm = find(S8_col~=0, 1, 'first');
                    if hm < 2 
                        check_heat = check_heat+ 1;
                    end
                   
               end
               norms_Avg = 0;
           min_check = abs(min(ds_all(:))-ds_pre_min); 
           min_check_diff = abs(ds_pre(i,1)-ds_pre_min); 
           ds_pre_1 = ds_pre;
          
           
           % ds_all input
           
           [row,col,value] = find(ds_all~=0);
                           
            if show_output == 43

                q_imgg = imread(char(qimg_path));
                db_imgg = imread(char(db_img));

                qqq_img = q_imgg;
                dbb_img = db_imgg;
            end
            
            
            for iii = 1:1
                if ~isempty(row) && ~isempty(imgg_mat_box_db) && ~isempty(imgg_mat_box_q)

                    box_var_db = [];
                    box_var_q = [];
                    AAsum =[];
                    norms= [];
                    Pslen_table = [];
                    Pslen_table_neg = [];
                    Pslen_table_43 = [];
                    %subplot(2,3,1);clf
                    %  subplot(2,3,2);clf
                    for jjj=1:length(row) %Top_boxes

                        %Query -> Row and DB -> DB1 DB2 DB3 DB4 DB5 DB6 DB7
                        %DB8
                        %



                        related_Box_dis_top = x_q_feat_ds_all(1,col(jjj));
                        related_Box_dis = x_q_feat_ds_all(row(jjj)+1,col(jjj));   

                        related_Box_q = x_q_feat_ids_all(row(jjj)+1,col(jjj));
                        related_Box_db = col(jjj);

                        q_size = x_q_feat_box_q(1,3)*(x_q_feat_box_q(1,4));  % wrong size, 3 se multiply howa howa hai
                        db_size = x_q_feat_box_db(1,3)*(x_q_feat_box_db(1,4));



                        bb_db = imgg_mat_box_db(related_Box_db,1:4); % Fix sized, so es ko 50 waly ki zarorat nai hai                      
                        box_var_db_i = [bb_db (bb_db(3)+bb_db(1))/2 (bb_db(4)+bb_db(2))/2] ;
                        box_var_db = [box_var_db ; box_var_db_i];

                        % ye query k sath hona chahihye
                        if  size(imgg_mat_box_q,1) < related_Box_q
                           bb_q = imgg_mat_box_q(1,1:4); % Fix sized, so es ko 50 waly ki zarorat nai hai
                        else
                          bb_q = imgg_mat_box_q(related_Box_q,1:4); % Fix sized, so es ko 50 waly ki zarorat nai hai
                        end

                        box_var_q_i = [bb_q (bb_q(3)+bb_q(1))/2 (bb_q(4)+bb_q(2))/2] ;
                        box_var_q = [box_var_q ; box_var_q_i];

                        q_width_height = (bb_q(1,3)*bb_q(1,4))/(q_size);
                        db_width_height = (bb_db(1,3)*bb_db(1,4))/(db_size);

                        exp_related_Box_dis = exp(-1.*related_Box_dis);%/ds_box_all_sum;
                        

                        sum_distance = ds_pre(1,1)+related_Box_dis-min(ds_all(:));
                        exp_sum_distance = exp(-1.*sum_distance); %*exp_related_Box_dis;
                        
                        exp_q_width_height = exp(-1.*(q_width_height));
                        exp_db_width_height = exp(-1.*(db_width_height));



                        Pslen_current = [single(related_Box_q) single(bb_q(1,3:4)) single(related_Box_db) single(bb_db(1,3:4)),...
                            related_Box_dis_top exp(-1.*related_Box_dis_top)/exp_ds_pre_sum related_Box_dis,...
                            exp(-1.*related_Box_dis)/exp_ds_pre_sum single(bb_q(1,3)/bb_db(1,3)) single(bb_q(1,4)/bb_db(1,4)) ,...
                            exp_related_Box_dis exp_q_width_height exp_db_width_height, ...
                            exp_relative_diff*exp_sum_distance*exp_q_width_height*exp_db_width_height,...
                            q_width_height db_width_height relative_diff] ;

                        % Pslen_Current :  
                        % 
                        % [1-3] Querybox                    
                        % Querybox# w h
                        % [4-6] DB box
                        % DBbox#    w h
                        % [7]   related_Box_dis_top
                        % box -> DB images
                        % [8]   exp(-1.*related_Box_dis_top)/exp_ds_pre_sum
                        % first row has box of query to full DB images
                        % exp_ds_pre_sum is the sum of all db values       
                        % exp_ds_pre = exp(-1.*ds_pre); (previous distances)
                        % exp_ds_pre_sum = sum(exp_ds_pre);
                        % [9]   related_Box_dis
                        %       box to box matched distance
                        % [10]  exp(-1.*related_Box_dis)/exp_ds_pre_sum
                        % box to box matched distance / the pre sum
                        % [11]  single(bb_q(1,3)/bb_db(1,3)) 
                        % width of query vs width of db
                        % [12]  single(bb_q(1,4)/bb_db(1,4))]
                        % height of query vs heigth of the db image
                        % [13]  single(bb_q(1,4)/bb_db(1,4))]
                        % width_height P of each box



                        Pslen_table = [Pslen_table ; Pslen_current]; 

                    end

               end


               prob_ds_All = 1;
               
               mean_min_top = exp(-1.*max(x_q_feat_ds_all(1,:)));

               if ~isempty(Pslen_table)
                                  min_ds_pre_all = [min_ds_pre_all; ds_pre(1,1) ds_pre(i,1) min(ds_all(:)) mean(ds_all(:)) min(x_q_feat_ds_all(1,:)) mean(x_q_feat_ds_all(1,:)) mean(Pslen_table(:,17)) mean(Pslen_table(:,18)) ];

               q_width_height_0 = Pslen_table(:,17);
               q_width_height_1 = Pslen_table(:,17);
               q_width_height_2 = Pslen_table(:,17);

               q_width_height_0(q_width_height_0>1) = 0;
               q_width_height_2(q_width_height_2<2) = 0;
               q_width_height_1(q_width_height_1>2) = 0;
               q_width_height_1(q_width_height_1<1) = 0;

               db_width_height_0 = Pslen_table(:,18);
               db_width_height_1 = Pslen_table(:,18);
               db_width_height_2 = Pslen_table(:,18);

               db_width_height_0(db_width_height_0>1) = 0;
               db_width_height_2(db_width_height_2<2) = 0;
               db_width_height_1(db_width_height_1>2) = 0;
               db_width_height_1(db_width_height_1<1) = 0;


               prob_ds_pre_sum = exp_ds_pre(i,1)/exp_ds_pre_sum;

               prob_ds_All = prob_ds_pre_sum*sum(Pslen_table(:,16));

               end
               ds_pre_diff(i,2) = prob_ds_All;
               ds_pre_gt(i,2) = prob_ds_All;
               ds_pre_gt(i,3) = D_diff;

        
           
          
           D_diff = D_diff+prob_ds_All-mean_min_top;%;%-mean(ds_pre_diff);

           prob_q_db(i,1) = D_diff;
           ds_pre_1(i,1) = D_diff;    
               
              

            end
           
           
          
           
           
          
             
           ds_new_top(i,1) = abs(D_diff);
           
         Pslen_table = [];

         ds_all = [];
        end
        
        %  SLEN_top(i,1) = i; SLEN_top(i,2) = aa;
        
%         Slot_group = 5;
%         idss = ids;
% 
%         for i=1:total_top/Slot_group
%             slot_Start = i*Slot_group-Slot_group;
%             slot_till = i*Slot_group;
%             [C c_i] = sortrows(ds_new_top(1+slot_Start:slot_till,1));
%             for j=1:Slot_group
%                 idss(j+slot_Start,1) = ids(c_i(j,1)+slot_Start);
%             end
% 
% 
%         end
          gmm_gt = [gmm_gt ; ds_pre_gt];
          
          X2 = ds_pre_gt(:,2:3);
          T = array2table(X2,...
    'VariableNames',{'VarName2','VarName3'});
          yfit = trainedModel.predictFcn(T) ;
        
        %[C c_i] = sortrows(ds_new_top);
        [C c_i] = sortrows(yfit,'descend');

        idss = ids;
        inegatifss = inegatif_i;
        for i=1:total_top
            idss(i,1) = ids(c_i(i,1));
            inegatifss(i,1) = inegatif_i(c_i(i,1));

        end
         if show_output == 3

                subplot(2,6,1); imshow(imread(char(qimg_path))); %q_img
                db_imgo1 = strcat(dataset_path,'/images/', db.dbImageFns{ids(1,1),1});  
                db_imgo2 = strcat(dataset_path,'/images/', db.dbImageFns{ids(2,1),1});  
                db_imgo3 = strcat(dataset_path,'/images/', db.dbImageFns{ids(3,1),1});  
                db_imgo4 = strcat(dataset_path,'/images/', db.dbImageFns{ids(4,1),1});  
                db_imgo5 = strcat(dataset_path,'/images/', db.dbImageFns{ids(5,1),1});  
                db_img1 = strcat(dataset_path,'/images/', db.dbImageFns{idss(1,1),1});  
                db_img2 = strcat(dataset_path,'/images/', db.dbImageFns{idss(2,1),1});  
                db_img3 = strcat(dataset_path,'/images/', db.dbImageFns{idss(3,1),1});
                db_img4 = strcat(dataset_path,'/images/', db.dbImageFns{idss(4,1),1});
                db_img5 = strcat(dataset_path,'/images/', db.dbImageFns{idss(5,1),1});
                
                subplot(2,6,2); imshow(imread(char(db_imgo1))); %
                aa = strcat(string(ds_pre(1,1)));title(aa)

                subplot(2,6,3); imshow(imread(char(db_imgo2))); %
                aa = strcat(string(ds_pre(2,1)));title(aa)

                subplot(2,6,4); imshow(imread(char(db_imgo3))); %
                aa = strcat(string(ds_pre(3,1)));title(aa)

                subplot(2,6,5); imshow(imread(char(db_imgo4))); %
                aa = strcat(string(ds_pre(4,1)));title(aa)

                subplot(2,6,6); imshow(imread(char(db_imgo5))); %
                aa = strcat(string(ds_pre(5,1)));title(aa)

                
                subplot(2,6,8); imshow(imread(char(db_img1))); %
                aa = strcat(string(ds_new_top(1,1)), '->', string(prob_q_db(1,1)));title(aa)
                subplot(2,6,9); imshow(imread(char(db_img2))); %
                aa = strcat(string(ds_new_top(2,1)), '->', string(prob_q_db(1,1)));title(aa)
                subplot(2,6,10); imshow(imread(char(db_img3))); %
                aa = strcat(string(ds_new_top(3,1)), '->', string(prob_q_db(1,1)));title(aa)
                subplot(2,6,11); imshow(imread(char(db_img4))); %
                aa = strcat(string(ds_new_top(4,1)), '->', string(prob_q_db(1,1)));title(aa)
                subplot(2,6,12); imshow(imread(char(db_img5))); %
                aa = strcat(string(ds_new_top(5,1)), '->', string(prob_q_db(1,1)));title(aa)

                %fprintf( '==>> %f %f %f %f %f \n',c_i(1,1), c_i(2,1),c_i(3,1), c_i(4,1) ,c_i(5,1));

         end
         
          if show_output == 33

                subplot(2,6,1); imshow(imread(char(qimg_path))); %q_img
                db_imgo1 = strcat(dataset_path,'/images/', db.dbImageFns{idss(1,1),1});  
                db_imgo2 = strcat(dataset_path,'/images/', db.dbImageFns{idss(2,1),1});  
                db_imgo3 = strcat(dataset_path,'/images/', db.dbImageFns{idss(3,1),1});
                db_imgo4 = strcat(dataset_path,'/images/', db.dbImageFns{idss(4,1),1});
                db_imgo5 = strcat(dataset_path,'/images/', db.dbImageFns{idss(5,1),1});
                db_img1 = strcat(dataset_path,'/images/', db.dbImageFns{idss(6,1),1});  
                db_img2 = strcat(dataset_path,'/images/', db.dbImageFns{idss(7,1),1});  
                db_img3 = strcat(dataset_path,'/images/', db.dbImageFns{idss(8,1),1});  
                db_img4 = strcat(dataset_path,'/images/', db.dbImageFns{idss(9,1),1});  
                db_img5 = strcat(dataset_path,'/images/', db.dbImageFns{idss(10,1),1});  
                
                subplot(2,6,2); imshow(imread(char(db_imgo1))); %
                aa = strcat(string(ds_new_top(1,1)), '->', string(prob_q_db(1,1)));title(aa)
                subplot(2,6,3); imshow(imread(char(db_imgo2))); %
                aa = strcat(string(ds_new_top(2,1)), '->', string(prob_q_db(2,1)));title(aa)
                subplot(2,6,4); imshow(imread(char(db_imgo3))); %
                aa = strcat(string(ds_new_top(3,1)), '->', string(prob_q_db(3,1)));title(aa)
                subplot(2,6,5); imshow(imread(char(db_imgo4))); %
                aa = strcat(string(ds_new_top(4,1)), '->', string(prob_q_db(4,1)));title(aa)
                subplot(2,6,6); imshow(imread(char(db_imgo5))); %
                aa = strcat(string(ds_new_top(5,1)), '->', string(prob_q_db(5,1)));title(aa)
                subplot(2,6,8); imshow(imread(char(db_img1))); %
                aa = strcat(string(ds_new_top(6,1)), '->', string(prob_q_db(6,1)));title(aa)
                subplot(2,6,9); imshow(imread(char(db_img2))); %
                aa = strcat(string(ds_new_top(7,1)), '->', string(prob_q_db(7,1)));title(aa)
                subplot(2,6,10); imshow(imread(char(db_img3))); %
                aa = strcat(string(ds_new_top(8,1)), '->', string(prob_q_db(8,1)));title(aa)
                subplot(2,6,11); imshow(imread(char(db_img4))); %
                aa = strcat(string(ds_new_top(9,1)), '->', string(prob_q_db(9,1)));title(aa)
                subplot(2,6,12); imshow(imread(char(db_img5))); %
                aa = strcat(string(ds_new_top(10,1)), '->', string(prob_q_db(10,1)));title(aa)
%                fprintf( '==>> %f %f %f %f %f %f %f %f %f %f \n',c_i(1,1), c_i(2,1),c_i(3,1), c_i(4,1) ,c_i(5,1), c_i(6,1), c_i(7,1),c_i(8,1), c_i(9,1) ,c_i(10,1));

         end
        
        
        iTestSample
        %% LEO END
            
        
       
        
        
        
        
        numReturned= length(ids);
        assert(numReturned<=nTop); % if your searcher returns fewer, it's your fault
        
        thisRecall= cumsum( isPos(iTest, idss) ) > 0; % yahan se get karta hai %db.cp (close position)
        recalls(iTestSample, :)= thisRecall( min(ns, numReturned) );
        
        thisRecall1= cumsum( isPos(iTest, ids) ) > 0; % yahan se get karta hai %db.cp (close position)
        recalls_ori(iTestSample, :)= thisRecall1( min(ns, numReturned) );
        printRecalls(iTestSample)= thisRecall(printN);
        
        thisRecall_idx = find(thisRecall~=0, 1, 'first');
        thisRecall1_idx = find(thisRecall1~=0, 1, 'first');
        fprintf('PLEN Recall: %i and Original Recall: %i \n',thisRecall_idx, thisRecall1_idx );
        if ~(isempty(thisRecall_idx) && isempty(thisRecall1_idx))
          if  ((thisRecall_idx-thisRecall1_idx) > 0 && thisRecall1_idx < 4) 
               fprintf('iTestSample: %i \n',iTestSample);
     
          end
        end
        if show_output == 45
               fprintf('iTestSample: %i \n',iTestSample);
               figure;
               subplot(2,2,1);
               plot(box_var_db(c_i(1,1),:), 'ro-'); hold on
               plot(box_var_db(c_i(2,1),:), 'ro-'); hold on
               plot(box_var_db(c_i(3,1),:), 'ro-'); hold on
               plot(box_var_db(c_i(4,1),:), 'ro-'); hold on
               plot(box_var_db(c_i(5,1),:), 'ro-'); hold on
               plot(box_var_db(c_i(thisRecall_idx,1),:), 'go-'); hold on
               plot(box_var_db(thisRecall1_idx,:), 'bo-'); hold on
                subplot(2,2,2);imshow(imread(char(qimg_path)));

                 subplot(2,2,3);  
                 db_imgo1 = strcat(dataset_path,'/images/', db.dbImageFns{ids(c_i(thisRecall_idx,1),1),1});  
                 imshow(imread(char(db_imgo1))); %
                  subplot(2,2,4);  
                 db_imgo2 = strcat(dataset_path,'/images/', db.dbImageFns{ids(thisRecall1_idx,1),1});  
                 imshow(imread(char(db_imgo2))); %
        end
        if thisRecall(1) == 0
          fprintf('iTestSample: %i \n',iTestSample);
  %           plot(ns, recalls(1:iTestSample,:), 'ro-',ns, recalls_ori(1:iTestSample,:), 'go-'); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none');

        end
       
    end  
    t= toc(evalProg);
    
    res= mean(printRecalls);
    relja_display('\n\trec@%d= %.4f, time= %.4f s, avgTime= %.4f ms\n', printN, res, t, t*1000/length(toTest));
    save('pslen-tokyo2tokto-vt-7.mat','x_q_feat_all');
    save('pslen-tokyo2tokto-GMM-87.mat','gmm_gt');

    relja_display('%03d %.4f\n', [ns(:), mean(recalls,1)']');
    
    rng(rngState);
end

   

function plot_mat(A)
lowestValue = min(A(A(:)>0));
highestValue = max(A(:));
imagesc(A);
cmap = jet(256);
colormap(cmap);
caxis(gca,[lowestValue-2/256, highestValue]);
% Make less than lowest value black:
cmap(1,:)=[0,0,0];
colormap(cmap)
caxis([-0.2 0.2]);
colorbar
end

function [mat_boxes,im, edge_image, hyt, wyd] = img_Bbox(db_img,model)
im= vl_imreadjpeg({char(db_img)},'numThreads', 12); 
I = uint8(im{1,1});
[bbox, E] =edgeBoxes(I,model);
[hyt, wyd] = size(im{1,1});
edge_image = uint8(E * 255);
bboxes=[];
gt=[111	98	25	101];

b_size = size(bbox,1); 
for ii=1:b_size
     bb=bbox(ii,:);
     square = bb(3)*bb(4);
     if square <2*gt(3)*gt(4)
        bboxes=[bbox;bb];
     end
end

mat_boxes = uint8(bboxes); 
end

function img = draw_boxx(I,bb)

%5bb=[bb(1) bb(2) bb(3)+bb(1) bb(4)+bb(2)];

%img = insertShape(I,'Rectangle',bb,'LineWidth',3);

%drawRectangle(image, Xmin, Ymin, width, height)
img = drawRectangle(I, bb(2), bb(1), bb(4), bb(3));

end
