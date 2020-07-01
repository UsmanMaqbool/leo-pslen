% Demo for Edge Boxes (please see readme.txt first).
clc;
addpath(genpath(pwd));

%% load pre-trained edge detection model and set opts (see edgesDemo.m)

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
    

%% detect Edge Box bounding box proposals (see edgeBoxes.m)
% Take two images (left image and right image)
I_r = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/images/03818/381899.552_3946607.103/VEGfuMbTsOSGRtqyw2LKdg_012_060.jpg';
I_l = '/home/leo/docker_ws/datasets/Test_247_Tokyo_GSV/query/00045.jpg';

im= vl_imreadjpeg({char(I_r)},'numThreads', 12); 

I = uint8(im{1,1});
[bbox, ~] =edgeBoxes(I,model);


[gtRes_l, dtRes_l] = calculate_gtRes_dtRes(bboxes_l);
[gtRes_r, dtRes_r] = calculate_gtRes_dtRes(bboxes_r);

%bboxes_l_sort = nms_box(dtRes_l, .1);
RGB_l = draw_rect_img(I_l, dtRes_l);
RGB_r = draw_rect_img(I_r, dtRes_r);




e8u_l = uint8(E_l * 255);
e8u_r = uint8(E_r * 255);

I_cropped_l = take_boxes(I_l, dtRes_l);
I_cropped_r = take_boxes(I_r, dtRes_r);
I_cropped_e8u_l = take_boxes(e8u_l, dtRes_l);
I_cropped_e8u_r = take_boxes(e8u_r, dtRes_r);


I_cropped_l_only = take_boxes_only(I_l, dtRes_l);
I_cropped_r_only = take_boxes_only(I_r, dtRes_r);
I_cropped_e8u_l_only = take_boxes_only(e8u_l, dtRes_l);
I_cropped_e8u_r_only = take_boxes_only(e8u_r, dtRes_r);


figure, imshow(RGB_l)
figure, imshow(RGB_r)

figure, imshow(I_cropped_l)




% imwrite(I_l, "original_l.jpg")
% imwrite(I_r, "original_r.jpg")
% imwrite(e8u_l, "E_l.jpg")
% imwrite(e8u_r, "E_r.jpg")
% imwrite(I_cropped_l, "I_cropped_l.jpg")
% imwrite(I_cropped_r, "I_cropped_r.jpg")
% imwrite(I_cropped_e8u_l, "I_cropped_e8u_l.jpg")
% imwrite(I_cropped_e8u_r, "I_cropped_e8u_r.jpg")
% imwrite(I_cropped_l_only, "I_cropped_l_only.jpg")
% imwrite(I_cropped_r_only, "I_cropped_r_only.jpg")
% imwrite(I_cropped_e8u_l_only, "I_cropped_e8u_l_only.jpg")
% imwrite(I_cropped_e8u_r_only, "I_cropped_e8u_r_only.jpg")
% Mme8u = mean(e8u(:));
% e8u(e8u>200)  = 0;
% e8u(e8u<100)  = 0;

BW = im2bw(e8u,0.5);

%test_enc = encode_edges(e8u);

%save_edge('save_edge.txt', e8u);


function RGB = draw_rect_img(I, dtRes)
colorpool=['g','y','m','b','b','b','b','b','b','b','b','b','b','b','b'];
RGB = I;
for ii=1:5
    bb=dtRes(ii,:);
   % rectangle('Position',[bb(1) bb(2) bb(3) bb(4)],'edgecolor',colorpool(ii));
    RGB = insertShape(RGB,'rectangle',[bb(1) bb(2) bb(3) bb(4)],'LineWidth',3);

end
end

function I_cropped = take_boxes_only(I, dtRes)

[x y c] = size(I);

for ii=1:5
    bb=dtRes(ii,:);
    if ii == 1
        xx_1 = bb(1); yy_1 = bb(2); xx_2 = bb(1)+bb(3); yy_2 = bb(2)+bb(4);
    else
       if xx_1 > bb(1)
           xx_1 = bb(1);
       end
       if yy_1 > bb(2)
           yy_1 = bb(2);
       end
       if xx_2 < bb(1)+bb(3)
           yy_2 = bb(1)+bb(3);
       end
       if yy_2 < bb(2)+bb(4)
           yy_2 = bb(2)+bb(4);
       end 
    end
    
end
I_cropped = uint8(zeros(xx_2-xx_1,yy_2-yy_1,c));

I_cropped(1:yy_2-yy_1+1, 1:xx_2-xx_1+1,:) = I(yy_1:yy_2, xx_1:xx_2,:);
end


function I_cropped = take_boxes(I, dtRes)

[x,y,c] = size(I);
I_cropped = uint8(zeros(size(I)));
for ii=1:5
    
    bb=dtRes(ii,:)
    I_cropped(bb(2):bb(2)+bb(4), bb(1):bb(1)+bb(3), :) = I(bb(2):bb(2)+bb(4), bb(1):bb(1)+bb(3), :);

end
end

function save_edge(file_name, data_edge)
% open your file for writing
 fid = fopen(file_name,'wt');
 % write the matrix
 entry_count = 0;
 [h,w] = size(data_edge);
 if fid > 0
 
    for ii = 1:h
        for jj = 1:w
            
            if data_edge(ii,jj) ~= 0
                 fprintf(fid,'%u %u',entry_count,data_edge(ii,jj));
                  entry_count = 0;
            else
               entry_count = entry_count + 1;
            end
        end
    end     
 fclose(fid);
 end
end

function [gtRes,dtRes] = calculate_gtRes_dtRes(bboxes)
gt = bboxes;
[bul, I] = sort(gt(:,end),'descend');
gt = gt(I,:);
nLandmarks = min(20, size(gt,1));
nLandmarks
gt = gt(1:nLandmarks, :);
[gtRes,dtRes]=bbGt('evalRes',gt,double(bboxes),.4);

end

function en_edge = encode_edges(e8u)
[h,w] = size(e8u);
enc_edge = [];
entry_count = 0;
for ii = 1:h
    for jj = 1:w
          
        if e8u(ii,jj) ~= 0
             enc_edge = [enc_edge;entry_count e8u(ii,jj)];
             entry_count = 0;
        else
           entry_count = entry_count + 1;
        end
    end
end
en_edge = enc_edge;
end


function top = nms_box(boxes, overlap)

% top = nms(boxes, overlap) 
% Non-maximum suppression.
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected detection.

if isempty(boxes)
  top = [];
else
  x1 = boxes(:,1);
  y1 = boxes(:,2);
  x2 = boxes(:,3);
  y2 = boxes(:,4);
  s = boxes(:,end);
  area = (x2-x1+1) .* (y2-y1+1);

  [vals, I] = sort(s);
  pick = [];
  while ~isempty(I)
    last = length(I);
    i = I(last);
    pick = [pick; i];
    suppress = [last];
    for pos = 1:last-1
      j = I(pos);
      xx1 = max(x1(i), x1(j));
      yy1 = max(y1(i), y1(j));
      xx2 = min(x2(i), x2(j));
      yy2 = min(y2(i), y2(j));
      w = xx2-xx1+1;
      h = yy2-yy1+1;
      if w > 0 && h > 0
        % compute overlap 
        o = w * h / area(j);
        if o > overlap
          suppress = [suppress; pos];
        end
      end
    end
    I(suppress) = [];
  end  
  top = boxes(pick,:);
end
end

%e8u_80(e8u_80<160) = 0;
%blankimage(:,:,2) = e8u_80;
% e8u_80 = e8u;
% e8u_80(e8u_80>160) = 0;
% e8u_80(e8u_80<80) = 0;
% blankimage(:,:,3) = e8u_80;
% imshow(e8u)
% imshow(blankimage)
% sz = net.Layers(1).InputSize 

%sz =

 %  224   224     3

%I = blankimage;
%I = I(1:sz(1),1:sz(2),1:sz(3)); 
%label = classify(net, I) 
