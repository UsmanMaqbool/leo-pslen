% https://ch.mathworks.com/help/stats/examples/classification.html#d120e3660

clear all, close all , clc; 
load fisheriris
f = figure;
gscatter(meas(:,1), meas(:,2), species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
N = size(meas,1);

lda = fitcdiscr(meas(:,1:2),species);
ldaClass = resubPredict(lda);
ldaResubErr = resubLoss(lda)
figure
ldaResubCM = confusionchart(species,ldaClass);


figure(f)
bad = ~strcmp(ldaClass,species);
hold on;
plot(meas(bad,1), meas(bad,2), 'kx');
hold off;

[x,y] = meshgrid(4:.1:8,2:.1:4.5);
x = x(:);
y = y(:);
j = classify([x y],meas(:,1:2),species);
gscatter(x,y,j,'grb','sod')

qda = fitcdiscr(meas(:,1:2),species,'DiscrimType','quadratic');
qdaResubErr = resubLoss(qda)

cp = cvpartition(species,'KFold',10)

cvlda = crossval(lda,'CVPartition',cp);
ldaCVErr = kfoldLoss(cvlda)
cvqda = crossval(qda,'CVPartition',cp);
qdaCVErr = kfoldLoss(cvqda)



nbGau = fitcnb(meas(:,1:2), species);
nbGauResubErr = resubLoss(nbGau)
nbGauCV = crossval(nbGau, 'CVPartition',cp);
nbGauCVErr = kfoldLoss(nbGauCV)

labels = predict(nbGau, [x y]);
gscatter(x,y,labels,'grb','sod')

nbKD = fitcnb(meas(:,1:2), species, 'DistributionNames','kernel', 'Kernel','box');
nbKDResubErr = resubLoss(nbKD)
nbKDCV = crossval(nbKD, 'CVPartition',cp);
nbKDCVErr = kfoldLoss(nbKDCV)

labels = predict(nbKD, [x y]);
gscatter(x,y,labels,'rgb','osd')
t = fitctree(meas(:,1:2), species,'PredictorNames',{'SL' 'SW' });

[grpname,node] = predict(t,[x y]);
gscatter(x,y,grpname,'grb','sod')


view(t,'Mode','graph');

dtResubErr = resubLoss(t)

cvt = crossval(t,'CVPartition',cp);
dtCVErr = kfoldLoss(cvt)

resubcost = resubLoss(t,'Subtrees','all');
[cost,secost,ntermnodes,bestlevel] = cvloss(t,'Subtrees','all');
plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
figure(gcf);
xlabel('Number of terminal nodes');
ylabel('Cost (misclassification error)')
legend('Cross-validation','Resubstitution')

[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
plot([0 20], [cutoff cutoff], 'k:')
plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')
hold off

cost(bestlevel+1)