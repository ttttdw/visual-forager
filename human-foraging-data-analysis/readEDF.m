clear all; close all; clc;

subjectName = '';
folder = [subjectName];

datafileList = dir(fullfile(folder, '*.edf'));
fileDates = [datafileList.datenum];

[~, sortIndex] = sort(fileDates);
datafileList = datafileList(sortIndex);

eventlist = [];
for i = 1:length(datafileList)
    fullFileName = fullfile(folder, datafileList(i).name);
    edf0 = Edf2Mat(fullFileName);
    eventlist = [eventlist, edf0.RawEdf.FEVENT];
end

save([subjectName '/eventlist'], 'eventlist')