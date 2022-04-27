% Roman Ramirez, rr8rk@virginia.edu
% BME 4315/6315, Module 4 PCA

close all;
clear;

colors = [
    0 0 0 % black
    1 0.7 0 % orange
    1 0 0 % red
    0 0 1 % blue
    1 0 1 % pink
    0.5 0 0.5 % purple
    0 0.50 0 % green
    0 1 0 % green
    1 1 0.2 % yellow
    1 0.5 0.5 % pink
    ];
markers = {
    'o'
    '^'
    'square'
    'pentagram'
    'diamond'
    'v'
    };


%% READING THE FILE
raw = readtable('data/Alz_multivariate_dat.xlsx');

% the column in Excel is DX
cat_column = 4 * 26 + 24;

data_cat = raw(:,cat_column+1:end);
data_table = raw(:,1:cat_column);
data_matrix = data_table{:,:};
data_matrix_z = zscore(data_matrix, 0, 1);

clearvars cat_column

%% EXPERIMENTAL CONDITITIONS AND CLEANING

male = {0 1};

% proteins
variables_raw = raw.Properties.VariableNames(1:end-3);
variables = strings(size(variables_raw));

for i=1:length(variables_raw)

    foo = variables_raw{i};
    foo(foo == '_') = ' ';
    variables(i) = foo;

end

variables = convertStringsToChars(variables);

clearvars foo


% observations
patient_nums = 1:1:height(raw);

[n_observations n_variables] = size(data_matrix_z);

observations_male = data_cat{:,1};
observations_genotype = data_cat{:,2};
observations_diagnosis = data_cat{:,3};

males = unique(observations_male);
genotypes = unique(observations_genotype);
diagnoses = unique(observations_diagnosis);

%% PCA ANALYSIS

[coefficients,scores,latent,tsquared,explained,mu] = pca(data_matrix_z);

pcaFigure = figure(2);

% Percentage of variance explained by each PC
pcaVarianceFigure = figure(2);
plot(0:length(explained),[0 cumsum(explained)'],'linestyle','-','linewidth',2,'color','k','marker','o','markersize',8,'markerfacecolor','k','markeredgecolor','k');
set(gca,'XLim',[0 length(explained)],'YLim',[0 100],'Xtick',0:length(explained),'Xticklabel',0:length(explained),'Box','off');
title('Cumulative variance explained by PCA');
xlabel('Principal components (PCs)');
ylabel('% Variance explained');

saveas(pcaVarianceFigure, 'figures/pcaVarianceFigure', 'png');

%% PCA Score plot (with marker color/size)

numFigs = 9;

for counterX = 3:3% 1:numFigs
    
    pcaScoreFigure = figure;

    for counterY = 1:9
        subplot(3,3,counterY);
        hold on;
        
        h = gscatter(scores(:,counterX),scores(:,counterY),observations_diagnosis);
        row = 0;
        plot([-max(abs([min(scores(:,1)) max(scores(:,1))]))-0.5 max(abs([min(scores(:,1)) max(scores(:,1))]))+0.5],[0 0],'linestyle','-','linewidth',1,'color','k');
        plot([0 0],[-max(abs([min(scores(:,counterY)) max(scores(:,counterY))]))-0.5 max(abs([min(scores(:,counterY)) max(scores(:,counterY))]))+0.5],'linestyle','-','linewidth',1,'color','k');
        % set(gca,'fontsize',20);
        title('PCA score plot');
        xlabel(['PC' num2str(counterX)]);
        ylabel(['PC' num2str(counterY)]);
        legend('off');

        hold off
    end
    
    saveas(pcaScoreFigure, ['figures/pcaScoreFigure' num2str(counterX)], 'png')
end

%% DATA CLEANING: IMPAIRED

data_matrix_z_impaired = data_matrix_z(string(observations_diagnosis) == "Impaired", :);

%% WIP: CLUSTERING ANALYSIS

cluster1 = clustergram(data_matrix,'Cluster',3,'Symmetric',true,'Colormap',redblue(150),...
    'DisplayRange',3,'RowPDist','correlation','ColumnPDist','euclidean','Linkage','complete',...
    'RowLabels',observations_diagnosis); 


%% PLSR

% Response variable (72hr relative viability)
clear Y;
% Y = data_table{:, find(string(data_table.Properties.VariableNames) == "tau")};
% observations: tau and impaired
Y = data_matrix_z_impaired(:, string(data_table.Properties.VariableNames) == "tau");
% features: numerical data
X = data_matrix_z_impaired(:, string(data_table.Properties.VariableNames) ~= "tau");

%feature: numerical
%tau column

clear TSS;
clear PLSR_XLoading PLSR_YLoading PLSR_XScore PLSR_YScore PLSR_yfit;
clear Rsquare Qsquare;

TSS = sum((Y-mean(Y)).^2);
ncomp = length(Y)-1;
clear XLoading YLoading XScore YScore BETA PCTVAR MSE stats;
[XLoading,YLoading,XScore,YScore,BETA,PCTVAR,MSE,stats] = plsregress(X,Y,ncomp,'cv',length(Y));

% Prediction accuracy (leave-one-out cross validation)
Qsquare = [0 1-length(Y)*MSE(2,2:end)/TSS];
% Performance
Rsquare = [0 cumsum(PCTVAR(2,:))];

r2q2Figure = figure(6); % R2 and Q2 evaluation
plot(0:ncomp,100*Rsquare,'-b','linewidth',2,'marker','o','markersize',5');
hold on;
plot(0:ncomp,100*Qsquare,'-r','linewidth',2,'marker','o','markersize',5');
set(gca,'YLim',[0 110],'Box','off','XTick',1:2:99);
title('R^2 and Q^2 Plots');
xlabel('PLS Component');
ylabel('% Variance Explained/Predicted');
legend({'R^2' 'Q^2'}, 'location', 'southeast');
grid on;
% set(gca,'fontsize',20);

saveas(r2q2Figure, 'figures/r2q2Figure', 'png')


%% LEAVE-ONE-OUT STRATEGY

varsNoTau = variables(string(variables) ~= "tau");
varsNoTau = varsNoTau(string(varsNoTau) ~= "p tau");


% optimum number of PLS component
optimized_ncomp = 4;

% Rerun the models using the optimized number of PLS components
clear new_yfit;
for element_id = 1:length(Y) % pick an element, remove it and predict it using the rest of the dataset
    clear new_X new_Y;
    if element_id == 1
        new_Y = Y(element_id+1:end);
        new_X = X(element_id+1:end,:);
    elseif element_id == length(Y)
        new_Y = Y(1:element_id-1);
        new_X = X(1:element_id-1,:);
    else
        new_Y = [Y(1:element_id-1); Y(element_id+1:end)];
        new_X = [X(1:element_id-1,:); X(element_id+1:end,:)];
    end
    [new_XLoading,new_YLoading,new_XScore,new_YScore,new_BETA,new_PCTVAR,new_MSE,new_stats] = plsregress(new_X,new_Y,optimized_ncomp);
    new_yfit(element_id) = [ones(size(X(element_id,:),1),1) X(element_id,:)]*new_BETA;
end

looFigure = figure(7); % Visualize correlation between measured and predicted responses (through leave-one-out cross-validation)
hold on;
row = 0;

corrQ2 = corrcoef(new_yfit(1:9)', Y(1:9));
r2 = corrQ2(1,2);

clearvars a
gscatter(Y(1:9), new_yfit(1:9))


plot([-8 8],[-8 8],'linestyle','-','linewidth',1,'color','k');
title(['Leave-one-out cross-validation, R^2=' num2str(r2)]);
xlabel('Measured viability');
ylabel('Predicted viability');
grid on;
hold off;

saveas(looFigure, 'figures/looFigure', 'png');

%% Calculate VIP (Variable Importance in Projection) Scores
sum1 = zeros(1,n_variables-1);
sum2 = 0;
clear SS Wnorm2;
for i = 1:optimized_ncomp
    SS(i) = (YLoading(i)^2)*(XScore(:,i)'*XScore(:,i));
end
for i = 1:optimized_ncomp
    sum2 = sum2 + SS(i);
    Wnorm2(i) = stats.W(:,i)'*stats.W(:,i);
end

clear VIP;
for counter = 1:n_variables-1
    for k = 1:optimized_ncomp
        sum1(counter) = sum1(counter) + SS(k)*stats.W(counter,k)^2/Wnorm2(k);
    end
    VIP(counter) = ((n_variables-1)*sum1(counter)/sum2)^0.5;
end



vipTable = table;
vipTable.varName = varsNoTau';
vipTable.vip = VIP(string(varsNoTau) ~= "p tau")';

vipTableSorted = sortrows(vipTable, 'vip', 'descend');


%%

% Plot VIP scores
vipFigure = figure(8);
hold on
bar(VIP);
set(gca,'XTick',1:length(varsNoTau),'XTickLabel',varsNoTau,'YLim',[0 3.6], 'fontsize', 6);
title('Variable Importance in Projection');
xlabel('Signals');
ylabel('VIP score');
xtickangle(45);
hold off

% Plot sorted VIP scores
vipSortedFigure = figure(9);
hold on
bar(vipTableSorted{:,2});
set(gca,'XTick',1:length(varsNoTau),'XTickLabel',vipTableSorted{:,1},'YLim',[0 3.6], 'fontsize', 6);
title('Sorted Variable Importance in Projection');
xlabel('Signals');
ylabel('VIP score');
xtickangle(45);
hold off

% Plot sorted VIP scores
top = 10;
vipSortedHighestFigure = figure(10);
hold on
bar(vipTableSorted{1:top,2});
set(gca,'XTick',1:10,'XTickLabel',vipTableSorted{1:top,1});
title(['Top ' num2str(top) ' Variables Importance in Projection']);
xlabel('Signals');
ylabel('VIP score');
xtickangle(45);
hold off

saveas(vipFigure, 'figures/vipFigure', 'png');
saveas(vipSortedFigure, 'figures/vipSortedFigure', 'png');
saveas(vipSortedHighestFigure, 'figures/vipSortedHighestFigure', 'png');