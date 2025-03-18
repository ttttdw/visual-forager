load("modelRadiusScore.mat")

id1s = [id1(:,1)];
g1 = ones(length(id1s),1);
id2s = [id2(:,1)];
g2 = ones(length(id2s),1);
ood1s = [ood1(:,1)];
g3 = ones(length(ood1s),1);

load("chanceRadiusScore.mat")

id1s = [id1s; id1(:,1)];
g1 = [g1;2*ones(length(id1(:,1)),1)];
id2s = [id2s; id2(:,1)];
g2 = [g2;2*ones(length(id2(:,1)),1)];
ood1s = [ood1s; ood1(:,1)];
g3 = [g3;2*ones(length(ood1(:,1)),1)];

avo = anova1(id1s,g1)