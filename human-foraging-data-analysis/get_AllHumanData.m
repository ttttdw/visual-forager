subjectFolders = dir('all_subjects/*');
subjectFolders = subjectFolders([subjectFolders.isdir]);
subjectFolders = subjectFolders(~ismember({subjectFolders.name}, {'.', '..'}));
condition = '';
%add_on_condition = [16,32,64,128]

totalHumanScore = [];
totalUpperBound = [];
totalClickCount = [];
totalOnscreenCount = [];

for i = 1:length(subjectFolders)
    subjectPath = fullfile('all_subjects', subjectFolders(i).name);
    
    disp(subjectFolders(i).name)

    %preprocess(subjectPath);
    
   
    [humanScore, upperBound] = fcn_getHumanscore(subjectPath, condition,add_on_condition);
    totalHumanScore = [totalHumanScore; humanScore];
    totalUpperBound = [totalUpperBound; upperBound];
    
    [clickCount, onscreenCount] = fcn_getHumanbehaviour(subjectPath, condition,add_on_condition);
    
    if isempty(totalClickCount)
        totalClickCount = clickCount;
        totalOnscreenCount = onscreenCount;
    else
        totalClickCount = cat(3, totalClickCount, clickCount);
        totalOnscreenCount = cat(3, totalOnscreenCount, onscreenCount);
    end
end

save(['totalHumanScore/totalHumanScore' condition '.mat'], 'totalHumanScore');
save(['totalUpperBound/totalUpperBound' condition '.mat'], 'totalUpperBound');
save(['totalClickCount/totalClickCount' condition '.mat'], 'totalClickCount');
save(['totalOnscreenCount/totalOnscreenCount' condition '.mat'], 'totalOnscreenCount');
