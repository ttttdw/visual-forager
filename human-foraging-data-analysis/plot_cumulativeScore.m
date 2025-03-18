legends = {'Humans', 'VF(ours)', 'FeatOnly', 'AvgVal', 'MaxVal', 'Chance'};

%% Draw ID1 cumulative score
figure;
cumulativeScore = readmatrix("fixation model/ID1/cumulativeScore.csv");
display(std(cumulativeScore(:,end)))
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
display(cumulativeScore(end))
p1 = plot(1:length(cumulativeScore), cumulativeScore,  '-ob', 'LineWidth', 3, 'MarkerSize', 6);
hold on
load('totalclicklistid1.mat')
cumulativeScore = NaN(length(totalClicklist), 20);
final_score = [];
for t=1:length(totalClicklist)
    clickList = totalClicklist{t};
    for c=1:length(clickList)
        cumulativeScore(t, c) = clickList(c).score;
    end
    final_score = [final_score, clickList(c).score];
end
display(mean(final_score/208))
display(std(final_score/208))
cumulativeScore = mean(cumulativeScore, 1, "omitnan") / 208;
p2 = plot(1:length(cumulativeScore), cumulativeScore, '-sr', 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/ID1/cumulativeScoreChance.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p3 = plot(1:length(cumulativeScore), cumulativeScore, '-sk', 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/ID1/cumulativeScoreAdd.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p4 = plot(1:length(cumulativeScore), cumulativeScore, '-s', 'Color', [0.5 0.5 0.5], 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/ID1/cumulativeScoreValue.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p5 = plot(1:length(cumulativeScore), cumulativeScore, '-s', 'Color', [0.8 0.8 0.8], 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/ID1/cumulativeScorePopularity.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p6 = plot(1:length(cumulativeScore), cumulativeScore, '-s', 'Color', [0.3 0.3 0.3], 'LineWidth', 3, 'MarkerSize', 6);
hold off
%% tune plot to be beautiful
set(gca, 'FontSize', 24)
figure.Position = [0, 0, 800, 600];
figure.InnerPosition = [0, 0, 800, 600];
xlabel('Click number', 'FontName', 'Arial', 'FontSize', 32);
ylabel('Norm. Score', 'FontName', 'Arial', 'FontSize', 32);
% grid on;
legend([p2, p1, p6, p4, p5, p3], legends, 'FontName', 'Arial', 'FontSize', 24, 'Location', 'northwest', 'Box', 'off');
set(gca,'box','off');
set(gca, 'TickDir', 'out', 'TickLength', [0.01, 0.01]);
set(gca, 'LineWidth', 3);
xticks([1, 5, 10, 15, 20]);
yticks([0.1, 0.3, 0.5, 0.7, 0.9])

%% Draw ID2 cumulative score
figure;
cumulativeScore = readmatrix("fixation model/ID2/cumulativeScore.csv");
display(std(cumulativeScore(:,end)))
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
display(cumulativeScore(end))
p1 = plot(1:length(cumulativeScore), cumulativeScore,  '-ob', 'LineWidth', 3, 'MarkerSize', 6);
hold on
load('totalclicklistid2.mat')
final_score = [];
cumulativeScore = NaN(length(totalClicklist), 20);
for t=1:length(totalClicklist)
    clickList = totalClicklist{t};
    for c=1:length(clickList)
        cumulativeScore(t, c) = clickList(c).score;
    end
    final_score = [final_score, clickList(c).score];
end
display(mean(final_score/108))
display(std(final_score/108))
cumulativeScore = mean(cumulativeScore, 1, "omitnan") / 108;
p2 = plot(1:length(cumulativeScore), cumulativeScore, '-sr', 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/ID2/cumulativeScoreChance.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p3 = plot(1:length(cumulativeScore), cumulativeScore, '-sk', 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/ID2/cumulativeScoreAdd.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p4 = plot(1:length(cumulativeScore), cumulativeScore, '-s', 'Color', [0.5 0.5 0.5], 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/ID2/cumulativeScoreValue.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p5 = plot(1:length(cumulativeScore), cumulativeScore, '-s', 'Color', [0.8 0.8 0.8], 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/ID2/cumulativeScorePopularity.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p6 = plot(1:length(cumulativeScore), cumulativeScore, '-s', 'Color', [0.3 0.3 0.3], 'LineWidth', 3, 'MarkerSize', 6);
hold off
%% tune plot to be beautiful
set(gca, 'FontSize', 24)
figure.Position = [0, 0, 800, 600];
figure.InnerPosition = [0, 0, 800, 600];
xlabel('Click number', 'FontName', 'Arial', 'FontSize', 32);
ylabel('Norm. Score', 'FontName', 'Arial', 'FontSize', 32);
% grid on;
% legend([p1, p2, p3, p4, p5, p6], {'Model', 'Humans', 'Chance', 'Weighted summation', 'Value first', 'Popularity first'}, 'FontName', 'Arial', 'FontSize', 18, 'Location', 'northwest', 'Box', 'off');
set(gca,'box','off');
set(gca, 'TickDir', 'out', 'TickLength', [0.01, 0.01]);
set(gca, 'LineWidth', 3);
xticks([1, 5, 10, 15, 20]);
yticks([0.1, 0.3, 0.5, 0.7, 0.9])

%% Draw OOD1 cumulative score
figure;
cumulativeScore = readmatrix("fixation model/OOD1/cumulativeScore.csv");
display(std(cumulativeScore(:,end)))
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
display(cumulativeScore(end))
p1 = plot(1:length(cumulativeScore), cumulativeScore,  '-ob', 'LineWidth', 3, 'MarkerSize', 6);
hold on
load('totalclicklistood1.mat')
cumulativeScore = NaN(length(totalClicklist), 20);
final_score = [];
for t=1:length(totalClicklist)
    clickList = totalClicklist{t};
    for c=1:length(clickList)
        cumulativeScore(t, c) = clickList(c).score;
    end
    final_score = [final_score, clickList(c).score];
end
cumulativeScore = mean(cumulativeScore, 1, "omitnan") / 80;
display(mean(final_score/80))
display(std(final_score/80))
p2 = plot(1:length(cumulativeScore), cumulativeScore, '-sr', 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/OOD1/cumulativeScoreChance.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p3 = plot(1:length(cumulativeScore), cumulativeScore, '-sk', 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/OOD1/cumulativeScoreAdd.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p4 = plot(1:length(cumulativeScore), cumulativeScore, '-s', 'Color', [0.5 0.5 0.5], 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/OOD1/cumulativeScoreValue.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p5 = plot(1:length(cumulativeScore), cumulativeScore, '-s', 'Color', [0.8 0.8 0.8], 'LineWidth', 3, 'MarkerSize', 6);

cumulativeScore = readmatrix("fixation model/OOD1/cumulativeScorePopularity.csv");
cumulativeScore = mean(cumulativeScore, 1, "omitnan");
p6 = plot(1:length(cumulativeScore), cumulativeScore, '-s', 'Color', [0.3 0.3 0.3], 'LineWidth', 3, 'MarkerSize', 6);
hold off
%% tune plot to be beautiful
set(gca, 'FontSize', 24)
figure.Position = [0, 0, 800, 600];
figure.InnerPosition = [0, 0, 800, 600];
xlabel('Click number', 'FontName', 'Arial', 'FontSize', 32);
ylabel('Norm. Score', 'FontName', 'Arial', 'FontSize', 32);
% grid on;
% legend([p1, p2, p3, p4, p5, p6], {'Model', 'Humans', 'Chance', 'Weighted summation', 'Value first', 'Popularity first'}, 'FontName', 'Arial', 'FontSize', 18, 'Location', 'northwest', 'Box', 'off');
set(gca,'box','off');
set(gca, 'TickDir', 'out', 'TickLength', [0.01, 0.01]);
set(gca, 'LineWidth', 3);
xticks([1, 5, 10, 15, 20]);
yticks([0.1, 0.3, 0.5, 0.7, 0.9])

