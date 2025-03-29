function [clickCount, onscreenCount] = fcn_getHumanbehaviour(subject, condition)
filename = [subject '/data.mat'];
load(filename);
clickCount = [];
onscreenCount = [];
idx = structfind(triallist, 'taskmode', condition);
%idx2 = structfind(triallist, 'values', add_on_condition);
%idx = intersect(idx1, idx2);

if isempty(idx)
    return
end

clickCount = zeros(1,20,length(idx));
onscreenCount = zeros(1,20,length(idx));

for i=1:length(idx)
    startIdx=structfind(eventlist,'message',['TRIAL_ON: ' num2str(idx(i))]);
    endIdx=structfind(eventlist,'message',['TRIAL_OFF: ' num2str(idx(i))]);
    trialeventlist = eventlist(startIdx:endIdx);
    clicklist = fcn_getClicklist(trialeventlist);
    clickcount = zeros(1,20);
    for j=1:1
        clickId = structfind(clicklist, 'message', ['Target ' num2str(j)]);
        clickcount(j, clickId) = 1;
    end
    popularity = triallist(idx(i)).popularity;
    %disp(['Popularity shape: ' mat2str(size(popularity))]);
    onscreenCount(:,:,i) = ones(1, 20) .* popularity';
    clickCount(:,:,i) = clickcount;
end
onscreenCount = onscreenCount - cumsum(clickCount,2);
