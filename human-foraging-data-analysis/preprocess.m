function preprocess(subject)
folder = subject;
datafileList = dir(fullfile(folder, '/*.mat'));
for i=1:length(datafileList)
    fullFileName = fullfile(folder, datafileList(i).name);
    load(fullFileName);
end

triallist = struct();
for i=1:length(trialStimulus)
    triallist(i).taskmode = trialStimulus{i,1}.taskMode;
    triallist(i).id = trialStimulus{i,1}.blockid;
    triallist(i).setsize = trialStimulus{i,1}.setSize;
    triallist(i).popularity = trialStimulus{i,1}.popularities;
    triallist(i).values = trialStimulus{i,1}.values;
end

frame = 1;
for i=1:length(trialStimulus)
    startIdx=structfind(eventlist,'message',['TRIAL_ON: ' num2str(i)]);
    endIdx=structfind(eventlist,'message',['TRIAL_OFF: ' num2str(i)]);
    trialeventlist = eventlist(startIdx:endIdx);
    triallist(i).frame = frame;
    while 1
        idx = structfind(trialeventlist, 'message', ['FRAME_OFF: ', num2str(frame)]);
        if isempty(idx)
            break
        end
        frame = frame +1; 
    end
end
filename = [subject '/triallist.mat'];
save(filename, 'triallist');
filename = [subject '/data.mat'];
save(filename, 'triallist', 'eventlist', 'trialStimulus', 'framePosition')