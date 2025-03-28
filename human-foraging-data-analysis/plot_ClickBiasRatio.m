modelclickCountPath = 'fixation model/ID2/clickPercentage.csv';
modelonscreenCountPath = 'fixation model/ID2/onscreenPercentage.csv';
chanceclickCountPath = 'chance/ID2/clickPercentage.csv';
chanceonscreenCountPath = 'chance/ID2/onscreenPercentage.csv';
humanclickCountPath = 'totalClickCount/totalClickCount3.mat';
humanonscreenCountPath = 'totalOnscreenCount/totalOnscreenCount3.mat';

modelclickCountData = load(modelclickCountPath);
modelonscreenCountData = load(modelonscreenCountPath);
chanceclickCountData = load(chanceclickCountPath);
chanceonscreenCountData = load(chanceonscreenCountPath);
humanclickCountData = load(humanclickCountPath);
humanonscreenCountData = load(humanonscreenCountPath);

totalClickCount = humanclickCountData.totalClickCount;
totalOnscreenCount = humanonscreenCountData.totalOnscreenCount;
[rows, cols, trials] = size(totalClickCount);
numGroups = trials / 10;
clickCountAvg = zeros(rows, cols, numGroups);
onscreenCountAvg = zeros(rows, cols, numGroups);

for i = 1:numGroups
    trialIndices = (i-1) * 10 + 1 : i * 10;
    clickCountAvg(:, :, i) = mean(totalClickCount(:, :, trialIndices), 3);
    onscreenCountAvg(:, :, i) = mean(totalOnscreenCount(:, :, trialIndices), 3);
end

clickAvg = mean(clickCountAvg, 3);
onscreenAvg = mean(onscreenCountAvg, 3);
clickProportionAvg = clickAvg ./ sum(clickAvg, 1);
onscreenProportionAvg = onscreenAvg ./ sum(onscreenAvg, 1);

humanclick = mean(clickProportionAvg, 2);
humanonscreen = mean(onscreenProportionAvg, 2);
modelclick = mean(modelclickCountData, 2);
modelonscreen = mean(modelonscreenCountData, 2);
chanceclick = mean(chanceclickCountData, 2);
chanceonscreen = mean(chanceonscreenCountData, 2);

human_overpick_underpick = (humanclick - humanonscreen) ./ (humanclick + humanonscreen);
model_overpick_underpick = (modelclick - modelonscreen) ./ (modelclick + modelonscreen);
chance_overpick_underpick = (chanceclick - chanceonscreen) ./ (chanceclick + chanceonscreen);

data = [human_overpick_underpick, model_overpick_underpick, chance_overpick_underpick];


figure;
figWidth = 15;
figHeight = figWidth / 2;

set(gcf, 'Units', 'Inches', 'Position', [1, 1, figWidth, figHeight]);

b = bar(data);

b(1).FaceColor = [1 0 0]; % Human
b(2).FaceColor = [0 0 1]; % Model
b(3).FaceColor = [0.7 0.7 0.7]; % Chance

xticks(1:size(data, 1));
xticklabels({'2vals', '4vals', '8vals', '16vals'});
ylabel("Click Bias Ratio", 'FontSize', 24);  % Reduced font size
ylim([-0.60 0.61]);

lgd = legend({'Humans', 'VF(ours)', 'Chance'}, 'FontSize', 40,'FontWeight','bold');
set(lgd, 'Orientation', 'horizontal');
set(lgd, 'Location', 'northoutside');
set(lgd, 'Box', 'off');
set(lgd, 'Position', [0.2, 0.9, 0.6, 0.05]); 

box off;
ax = gca;
ax.XAxis.LineWidth = 4.0;
ax.YAxis.LineWidth = 4.0;
set(gca, 'FontSize', 45);

hold on;
hLine = plot(xlim, [0 0], 'k', 'LineWidth', 1.25);
set(get(get(hLine, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold off;








 



