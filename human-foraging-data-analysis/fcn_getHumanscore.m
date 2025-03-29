function [humanScore, upperBound] = fcn_getHumanscore(subject, condition)
humanScore = [];
upperBound = [];
    
filename = [subject '/data.mat'];
load(filename);
idx = structfind(triallist, 'taskmode', condition);
%idx2 = structfind(triallist, 'values', add_on_condition);
%idx = intersect(idx1, idx2);

if isempty(idx)
    return
end

for i=1:length(idx)
    startIdx=structfind(eventlist,'message',['TRIAL_ON: ' num2str(idx(i))]);
    endIdx=structfind(eventlist,'message',['TRIAL_OFF: ' num2str(idx(i))]);
    trialeventlist = eventlist(startIdx:endIdx);
    popularity = triallist(idx(i)).popularity;
    values = trialStimulus{idx(i)}.values;
    rewards = [];
    for p=1:length(popularity)
        rewards = [ones(1,popularity(p))*values(p) rewards];
    end
    upperbound = sum(rewards(1:20));
    upperBound = [upperBound; upperbound];
    clicklist = fcn_getClicklist(trialeventlist);
    humanScore = [humanScore; clicklist(end).score];
end
