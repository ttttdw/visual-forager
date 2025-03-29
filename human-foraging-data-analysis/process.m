subjectFolders = dir('/*');
subjectFolders = subjectFolders([subjectFolders.isdir]);
subjectFolders = subjectFolders(~ismember({subjectFolders.name}, {'.', '..'}));

for i = 1:length(subjectFolders)
    subjectPath = fullfile(subjectFolders(i).name);
    
    preprocess(subjectPath);
end
    