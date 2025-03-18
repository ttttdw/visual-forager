load("modelFixationlist.mat")

FixationSaccade = [];
for t=1:length(condition2)
    fixationList = condition2{t};
    [~, fixationSaccade] = fcn_getSaccadelist(fixationList);
    FixationSaccade = [FixationSaccade, fixationSaccade];
end
for t=1:length(condition1)
    fixationList = condition1{t};
    [~, fixationSaccade] = fcn_getSaccadelist(fixationList);
    FixationSaccade = [FixationSaccade, fixationSaccade];
end
for t=1:length(condition3)
    fixationList = condition3{t};
    [~, fixationSaccade] = fcn_getSaccadelist(fixationList);
    FixationSaccade = [FixationSaccade, fixationSaccade];
end
saccades = FixationSaccade * 30 / 75 / 1080;
fixationSaccadeSizes = atan(saccades) * 180 / pi;
h1 = histogram(fixationSaccadeSizes, (0:30), 'Normalization','percentage')
h1.FaceAlpha = 0.3;
h1.FaceColor = 'b';
xline(mean(fixationSaccadeSizes), '--', 'LineWidth', 2, 'Color', 'k')
text(mean(fixationSaccadeSizes)-1, 25, 'Mean', 'Color', 'b', 'FontSize', 18, 'LineWidth', 2, 'Rotation', 90);
%xline(median(fixationSaccadeSizes),'--', 'Median', 'FontSize', 18, 'LineWidth', 2)
hold on
x = linspace(min(fixationSaccadeSizes), max(fixationSaccadeSizes), 100);
pdf = normpdf(x, mean(fixationSaccadeSizes), std(fixationSaccadeSizes));
p1 = plot(x, pdf*100, 'b-', 'LineWidth', 2);

load("modelFixationlistNoecc.mat")

FixationSaccade = [];
for t=1:length(condition2)
    fixationList = condition2{t};
    [~, fixationSaccade] = fcn_getSaccadelist(fixationList);
    FixationSaccade = [FixationSaccade, fixationSaccade];
end
for t=1:length(condition1)
    fixationList = condition1{t};
    [~, fixationSaccade] = fcn_getSaccadelist(fixationList);
    FixationSaccade = [FixationSaccade, fixationSaccade];
end
for t=1:length(condition3)
    fixationList = condition3{t};
    [~, fixationSaccade] = fcn_getSaccadelist(fixationList);
    FixationSaccade = [FixationSaccade, fixationSaccade];
end
saccades = FixationSaccade * 30 / 75 / 1080;
fixationSaccadeSizes = atan(saccades) * 180 / pi;
h2 = histogram(fixationSaccadeSizes, (0:30), 'Normalization','percentage')
h2.FaceAlpha = 0.3;
h2.FaceColor = [0.4 0.6 1];
xline(mean(fixationSaccadeSizes), '--', 'LineWidth', 2, 'Color', 'k')
text(mean(fixationSaccadeSizes)-1, 25, 'Mean', 'Color', [0.4 0.6 1], 'FontSize', 18, 'LineWidth', 2, 'Rotation', 90);
%xline(median(fixationSaccadeSizes),'--', 'Median', 'FontSize', 18, 'LineWidth', 2)
x = linspace(min(fixationSaccadeSizes), max(fixationSaccadeSizes), 100);
pdf = normpdf(x, mean(fixationSaccadeSizes), std(fixationSaccadeSizes));
p2 = plot(x, pdf*100, '-', 'Color', [0.4 0.6 1], 'LineWidth', 2);


load('fixationsaccade.mat')
saccades = FixationSaccade * 30 / 75 / 1080;
fixationSaccadeSizes = atan(saccades) * 180 / pi;
h3 = histogram(fixationSaccadeSizes, (0:30), 'Normalization','percentage')
h3.FaceAlpha = 0.3;
h3.FaceColor = 'r';
xline(mean(fixationSaccadeSizes), '--', 'LineWidth', 2, 'Color', 'k')
text(mean(fixationSaccadeSizes)+1, 25, 'Mean', 'Color', 'r', 'FontSize', 18, 'LineWidth', 2, 'Rotation', 90);
%xline(median(fixationSaccadeSizes),'--', 'Median', 'FontSize', 18, 'LineWidth', 2)
x = linspace(min(fixationSaccadeSizes), max(fixationSaccadeSizes), 100);
pdf = normpdf(x, mean(fixationSaccadeSizes), std(fixationSaccadeSizes));
p3 = plot(x, pdf*100, 'r-', 'LineWidth', 2);
hold off


%% tune plot to be beautiful
set(gca, 'FontSize', 18)
legend([p3, p1, p2], {'Humans', 'VF(ours)', 'VF w/o eccentricity'}, 'FontName', 'Arial', 'FontSize', 18, 'Location', 'northeast', 'Box', 'off', 'FontWeight', 'bold');
set(gca,'box','off');
set(gca, 'TickDir', 'out', 'TickLength', [0.01, 0.01]);
set(gca, 'LineWidth', 1.5);
ylabel('Proportion (%)', 'FontSize', 24)
xlabel('Saccade Size (degrees)', 'FontSize', 24)