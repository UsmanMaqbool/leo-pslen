
qq_img = draw_boxx(q_imgg,bb_q);%   q_RGB = insertShape(I,'Rectangle',imgg_mat_box_q(row(jjj),1:4),'LineWidth',3);
dd_img = draw_boxx(db_imgg,bb_db);%   q_RGB = insertShape(I,'Rectangle',imgg_mat_box_q(row(jjj),1:4),'LineWidth',3);

qqq_img = draw_boxx(qqq_img,bb_q);%   q_RGB = insertShape(I,'Rectangle',imgg_mat_box_q(row(jjj),1:4),'LineWidth',3);
dbb_img = draw_boxx(dbb_img,bb_db);%   q_RGB = insertShape(I,'Rectangle',imgg_mat_box_q(row(jjj),1:4),'LineWidth',3);


subplot(2,3,1); imshow(qq_img); %q_img
title({
[ 'Qbox wxh exp = ' num2str(exp_q_width_height) ]
[ 'sum-table: ' num2str(sum(Pslen_table_43(:,1))) ' neg-table = ' num2str(sum(Pslen_table_43(:,2))) ]
['related-Box-dis = ' num2str(related_Box_dis) ]
[ 'related-Box-dis-top = ' num2str(related_Box_dis_top) ]

});

subplot(2,3,2); imshow(dd_img); %

ori_top_current = strcat(string(ds_pre(i,1)), '->', string(related_Box_dis_top),' ->', string(related_Box_dis));

title({
['dbbox wxh exp = ' num2str(exp_db_width_height) ]
[ 'QBox->DBBox Distance = ' num2str(related_Box_dis-ds_pre(i,1)) ]
[ 'QBox->DBBox Distance TOP = ' num2str(related_Box_dis_top-ds_pre(i,1)) ]
});

subplot(2,3,3); hold on; plot(box_var_q(jjj,:), 'ro-'); 



% subplot(2,3,3); bar(norms); %q_img
subplot(2,3,4); imshow(qqq_img); %q_img
if ~isempty(Pslen_table)
title({
[ 'min ds-pre ' num2str(min(ds_pre(:))) ]
[ 'current ds-pre(i,1) = ' num2str(ds_pre(i,1)) ]
[ 'min ds-all ' num2str(min(ds_all(:))) ]
[ 'original-Pslen-table ' num2str(ds_pre(i,1)-min(Pslen_table(:,7))) ]
});
end
subplot(2,3,5); imshow(dbb_img); %
if ~isempty(Pslen_table_neg)
title({
[ 'Pslen-table-neg ' num2str(min(Pslen_table_neg(:,7))) ]
[ 'original-Pslen-table-neg ' num2str(ds_pre(i,1)-min(Pslen_table_neg(:,7))) ]
});
end
subplot(2,3,6); hold on; plot(box_var_db(jjj,:), 'ro-'); 








%% Output 4
 fprintf(' %f -> %f %f %f %f %f %f %f \n',ds_pre(i,1), D_diff, num_var_s5,sum_var_s5, mum_var_s5,sol_4,sol_5,sol_6);
 y = [ds_pre(i,1) D_diff sum(S8(:)) inegatif sum(S5(:)) mum_var_s5 num_var_s5 nnz_black_check];
 q_imgg = imread(char(qimg_path));
 subplot(2,3,1); imshow(q_imgg); %q_img
 ori_top_curr = strcat(string(ds_pre(i,1)), '->', string(related_Box_dis_top));
 title(ori_top_curr)

db_imgg = imread(char(db_img));
subplot(2,3,2); imshow(db_imgg); %
title(Pslen_table_sum)



subplot(2,3,3); h = heatmap(S8);
subplot(2,3,4); h = heatmap(y); % with plus is wokring
subplot(2,3,5); h = heatmap(S3);
              