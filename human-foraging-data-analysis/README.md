## Using Human Data for analysis
### Quick Start
You can access the human data we collect through the following links: https://drive.google.com/drive/folders/1bwghYYpO2i606OIEvk7UVgO4O_ePoB2m?usp=drive_link  and put all data in folder
[all_subjects](./all_subjects) .

Then, you can run matlab code [get_AllHumanData](./get_AllHumanData.m) and get all **ClickCount**, **OnscreenCount**, **HumanScore** and **Upperbound** data.

Or if you want a quick preview, you may simply use the data in folder [totalClickCount](./totalClickCount/), etc.

### Raw Data Processing
After a stimulu collecting human eye-movement data, you may get a folder like this:

```
subject1/
├──record'subjedctname''timestamp'.mat 
├──***.edf
├──***.edf.tmp

```

You should copy the stimulu document in the same folder, and run matlab code [readEDF](./readEDF.m).
Put all folders together in one folder like this:

```
all_subjects/
├──subject1/
|   ├──record'subjedctname''timestamp'.mat 
|   ├──***.edf
|   ├──***.edf.tmp
|   ├──stimulus**.mat
|   ├──eventlist.mat
├──subject2/
├──subject3/

```
Then run matlab code [process](./process.m), you will get a batch of data similar to our's. Then repeat the process in [Quick Start](#quick-start).

