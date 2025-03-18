function [saccadelist,saccade] = fcn_getSaccadelist(fixlist)
saccadelist = struct();
saccade = [];
for i=1:length(fixlist)-1
    v1 = [fixlist(i).posX fixlist(i).posY];
    v2 = [fixlist(i+1).posX fixlist(i+1).posY];
    saccade = [saccade, norm(v1-v2)];
end
% 
% keywords = ['blank', 'Target1', 'Target2', 'Target3', 'Target4', 'Distractor'];
% 
% saccades = [];
% for i=1:length(saccadelist)
%     saccades = [saccades saccadelist(i).saccade];
% end
% histogram(saccades)