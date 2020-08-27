
close all; clear all;
clc;
load('pslen-pitts2tokto-data-4096-v2');

aa = data(1).H;
HH = [];
YY = [];
for i = 1:size(data,2)
    XX = data(i).X';
    XX = reshape(XX,1,[]);
    HH = [HH ; data(i).pre data(i).H XX double(data(i).Y)];

end
Data = array2table(HH);
hypopts = struct('ShowPlots',false,'Verbose',0,'UseParallel',false);
poolobj = gcp;


% Decision tree
mdls = fitrensemble(Data,'HH112');

save('pslen-pitts2tokto-data-4096-v2-rf','mdls');
