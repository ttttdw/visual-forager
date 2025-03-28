function clicklist = fcn_getClicklist(trialeventlist)
%% for each trial, get its click list
clicklist = [];
sublist = struct();
idx = structfind(trialeventlist,'message','Click: Target 4');
if length(idx) == 1
    sublist(1).idx = idx;
    sublist(1).message = 'Target 4';
else
    for i=1:length(idx)
        sublist(i).idx = idx(i);
        sublist(i).message = 'Target 4';
    end
end
if ~isempty(idx)
    clicklist = sublist;
end
sublist = struct();
idx = structfind(trialeventlist,'message','Click: Target 3');
if length(idx) == 1
    sublist(1).idx = idx;
    sublist(1).message = 'Target 3';
else
    for i=1:length(idx)
        sublist(i).idx = idx(i);
        sublist(i).message = 'Target 3';
    end
end
if ~isempty(idx)
    if isempty(clicklist)
        clicklist = sublist;
    else
        clicklist = [clicklist, sublist];
    end
end
sublist = struct();
idx = structfind(trialeventlist,'message','Click: Target 2');
if length(idx) == 1
    sublist(1).idx = idx;
    sublist(1).message = 'Target 2';
else
    for i=1:length(idx)
        sublist(i).idx = idx(i);
        sublist(i).message = 'Target 2';
    end
end
if ~isempty(idx)
    if isempty(clicklist)
        clicklist = sublist;
    else
        clicklist = [clicklist, sublist];
    end
end
sublist = struct();
idx = structfind(trialeventlist,'message','Click: Target 1');
if length(idx) == 1
    sublist(1).idx = idx;
    sublist(1).message = 'Target 1';
else
    for i=1:length(idx)
        sublist(i).idx = idx(i);
        sublist(i).message = 'Target 1';
    end
end
if ~isempty(idx)
    if isempty(clicklist)
        clicklist = sublist;
    else
        clicklist = [clicklist, sublist];
    end
end
sublist = struct();
idx = structfind(trialeventlist,'message','Click: blank');
if length(idx) == 1
    sublist(1).idx = idx;
    sublist(1).message = 'blank';
else
    for i=1:length(idx)
        sublist(i).idx = idx(i);
        sublist(i).message = 'blank';
    end
end
if ~isempty(idx)
    if isempty(clicklist)
        clicklist = sublist;
    else
        clicklist = [clicklist, sublist];
    end
end
sublist = struct();
idx = structfind(trialeventlist,'message','Click: Distractor');
if length(idx) == 1
    sublist(1).idx = idx;
    sublist(1).message = 'Distractor';
else
    for i=1:length(idx)
        sublist(i).idx = idx(i);
        sublist(i).message = 'Distractor';
    end
end
if ~isempty(idx)
    if isempty(clicklist)
        clicklist = sublist;
    else
        clicklist = [clicklist, sublist];
    end
end
[~,index] = sort([clicklist.idx]);
clicklist = clicklist(index);
messageIndex = structfind(trialeventlist, 'codestring', 'MESSAGEEVENT');
for i=1:length(clicklist)
    id = clicklist(i).idx+1;
    xMessage = trialeventlist(clicklist(i).idx+1).message;
    while isempty(xMessage)
        id = id + 1;
        xMessage = trialeventlist(id).message;
    end
    xPos = erase(xMessage, 'Click X: ');
    xPos = str2double(xPos);
    clicklist(i).posX = xPos;
    yMessage = trialeventlist(id+1).message;
    while isempty(yMessage)
       id = id + 1;
       if id > length(trialeventlist)
           error('无法找到有效的Y坐标信息');
       end
       yMessage = trialeventlist(id).message;
    end
    yPos = erase(yMessage, 'Click Y: ');
    yPos = str2double(yPos);
    sMessage = trialeventlist(id+2).message;
    if ~ischar(sMessage) && ~isstring(sMessage)
        sMessage = char(sMessage);
    end
    %score = erase(sMessage, 'Score: ');
    score = erase(sMessage, 'Score: ');
    score = str2double(score);
    clicklist(i).score = score;
    clicklist(i).posY = yPos;
    mIndex = messageIndex;
    mIndex(mIndex < clicklist(i).idx) = [];
    fMessage = trialeventlist(mIndex(5)).message;
    if strcmp(fMessage, 'Shuffled')
        fMessage = trialeventlist(mIndex(6)).message;
    end
    frame = erase(fMessage, 'FRAME_OFF: ');
    frame = str2double(frame);
    clicklist(i).frame = frame;
    starttime = trialeventlist(clicklist(i).idx).sttime - trialeventlist(1).sttime;
    clicklist(i).starttime = starttime;
end