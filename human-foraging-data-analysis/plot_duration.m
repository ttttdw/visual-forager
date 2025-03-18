load("durationData")
keywords = {'T 1'; 'T 2'; 'T 3'; 'T 4';'T 1'; 'T 2'; 'T 3'; 'T 4';'T 1'; 'T 2'; 'T 3'; 'T 4';};
avg = [];
error = [];
sem = [];
for tmId = 1:3
Durations = durationData(((tmId-1)*4+1):tmId*4,:);
avg = [avg;mean(Durations,2,"omitnan")];
error = [error;std(Durations,0,2,"omitnan")];
sem = [sem;std(Durations,0,2,"omitnan") / sqrt(size(Durations,2))];
end

avg = [avg(5:12);avg(1:4)];
error = [error(5:12);error(1:4)];
sem = [sem(5:12);sem(1:4)];
% figure;
bar((1:12)-0.5, avg, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'none', 'BarWidth',0.3)
hold on

% 再绘制散点图和 errorbar
e = errorbar((1:12)-0.5, avg, sem, '.', 'LineWidth', 2, 'Color', 'k');
e.Marker = 'none';
e.MarkerSize = 15;
hold on

xlim([0,13])
ylim([0,0.4]) % 设置y轴范围为0到550ms
xticks((1:12)-0.5)
xticklabels(keywords)
hold off

%% tune plot to be beautiful
set(gca, 'FontSize', 24)
% figure.Position = [0, 0, 400, 2400];
% figure.InnerPosition = [0, 0, 400, 2400];
% grid on;
set(gca,'box','off');
set(gca, 'TickDir', 'out', 'TickLength', [0.01, 0.01]);
set(gca, 'LineWidth', 3);
yticks([0.1, 0.2, 0.3, 0.4, 0.9])
xlabel('Target', 'FontName', 'Arial', 'FontSize', 32);
ylabel('Fixation Duration (sec)', 'FontName', 'Arial', 'FontSize', 32);