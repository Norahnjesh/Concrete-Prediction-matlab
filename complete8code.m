% Loading the dataset
data = readtable('Concrete_Data.csv');

% Checking for missing values
disp('Missing Values Summary:');
disp(sum(ismissing(data)))

% Computing the correlation matrix
correlation_matrix = corr(table2array(data));

% Getting the number of variables
numVars = size(correlation_matrix, 1);

% Generating numeric labels for axes
numericLabels = string(1:numVars);

% Creating the heatmap
figure;
h = heatmap(numericLabels, numericLabels, correlation_matrix, ...
    'ColorbarVisible', 'on', 'Colormap', parula);

% Setting heatmap properties for visibility
h.CellLabelFormat = '%.2f';  % Display correlation values
h.FontSize = 12;
h.MissingDataColor = [1, 1, 1]; % White color for missing data
h.GridVisible = 'on'; % Ensure grid lines for clarity

% Displaying  the numeric labels legend
disp('Legend for Numeric Labels:');
for i = 1:numVars
    fprintf('%d -> %s\n', i, data.Properties.VariableNames{i});
end

% Defining  feature matrix (X) and target variable (Y)
X = table2array(data(:, 1:end-1)); % Features (all columns except last)
Y = table2array(data(:, end));     % Target (last column)

% Computing mean and standard deviation for each feature
mu = mean(X);  
sigma = std(X);

% Computing Z-scores
Z_scores = (X - mu) ./ sigma;

% Defining the threshold (e.g., 3 standard deviations from the mean)
threshold = 3;
outliers_z = abs(Z_scores) > threshold;

% Displaying the number of outliers per feature
disp('Number of outliers per feature (Z-Score Method):');
disp(sum(outliers_z));

% --- INTERQUARTILE METHOD ---
Q1 = quantile(X, 0.25);  % First quartile
Q3 = quantile(X, 0.75);  % Third quartile
IQR = Q3 - Q1;           % Interquartile range

% Defining the thresholds for outliers
lower_bound = Q1 - 1.5 * IQR;
upper_bound = Q3 + 1.5 * IQR;

% Identifying  outliers
outliers_iqr = (X < lower_bound) | (X > upper_bound);

% Displaying  number of outliers per feature
disp('Number of outliers per feature (Interquartile Method):');
disp(sum(outliers_iqr));

% Combining removal of outliers
outliers_combined = outliers_z | outliers_iqr;

% Counting  outliers in each column
disp('Total outliers per feature (Combined):');
disp(sum(outliers_combined));

% Removing outliers from both X and Y
X_clean = X(~any(outliers_combined, 2), :);
Y_clean = Y(~any(outliers_combined, 2), :);

% Displaying the new size of the dataset
disp('Original size:');
disp(size(X));
disp('Cleaned size:');
disp(size(X_clean));

% Saving the cleaned data to a new CSV file
newcleaned_data = array2table([X_clean Y_clean], 'VariableNames', data.Properties.VariableNames);
writetable(newcleaned_data, 'newConcrete_Cleaned.csv');
%%Question 2
%OLS regression model with outliers
Concrete_Data = readtable('Concrete_Data.csv');
X = Concrete_Data{:, 1:end-1}; % Features (all columns except last)
y = Concrete_Data{:, end};   
% Setting  random seed for reproducibility
rng(1);

% Getting the number of samples
numSamples = size(X, 1);

% Generating the  random indices
idx = randperm(numSamples);

% Defining  split index (80% training, 20% testing)
splitIdx = round(0.8 * numSamples);

% Training set
X_train = X(idx(1:splitIdx), :);
y_train = y(idx(1:splitIdx), :);

% Testing the set
X_test = X(idx(splitIdx+1:end), :);
y_test = y(idx(splitIdx+1:end), :);
% Fit OLS model on training data
lm = fitlm(X_train, y_train);

% Displaying  model summary
disp(lm)
% Make predictions on test set
y_pred = predict(lm, X_test);

% Compute Mean Squared Error (MSE)
mse_test = mean((y_test - y_pred).^2);
disp(['Test Mean Squared Error: ', num2str(mse_test)])

% Compute R-squared on test set
SS_total = sum((y_test - mean(y_test)).^2);
SS_residual = sum((y_test - y_pred).^2);
R_squared_test = 1 - (SS_residual / SS_total);
disp(['Test R-squared: ', num2str(R_squared_test)])

% Scatter plot of actual vs predicted values
figure;
scatter(y_test, y_pred, 'b', 'filled');
hold on;
plot([min(y_test) max(y_test)], [min(y_test) max(y_test)], 'r--', 'LineWidth', 2);
xlabel('Actual Compressive Strength');
ylabel('Predicted Compressive Strength');
title('Actual vs Predicted Compressive Strength');
grid on;
hold off;
%OLS regression model without outliers
% Confirming  the cleaned file is saved
disp('newCleaned data saved as Concrete_Cleaned.csv');
Cleaned_Concrete_Data = readtable('Cleaned_Concrete_Data.csv');
summary(Cleaned_Concrete_Data) 
% Computing summary statistics for each column
MinVals = min(Cleaned_Concrete_Data{:,:});
MaxVals = max(Cleaned_Concrete_Data{:,:});
MeanVals = mean(Cleaned_Concrete_Data{:,:});
MedianVals = median(Cleaned_Concrete_Data{:,:});
StdVals = std(Cleaned_Concrete_Data{:,:});
MissingVals = sum(ismissing(Cleaned_Concrete_Data)); % Count missing values

% Creating  a clean summary table
summaryTable = table(MinVals', MaxVals', MeanVals', MedianVals', StdVals', MissingVals', ...
    'VariableNames', {'Min', 'Max', 'Mean', 'Median', 'StdDev', 'MissingValues'}, ...
    'RowNames', Cleaned_Concrete_Data.Properties.VariableNames);

% Displaying  the summary table
disp(summaryTable)

% Saving  the summary table to CSV
writetable(summaryTable, 'Cleaned_Concrete_Data_Summary.csv', 'WriteRowNames', true);

disp('Summary table saved as Cleaned_Concrete_Data.csv');
X = Cleaned_Concrete_Data{:, 1:end-1}; % Features (all columns except last)
y = Cleaned_Concrete_Data{:, end};   
% Set random seed for reproducibility
rng(1);

% Getting  number of samples
numSamples = size(X, 1);

% Generating  random indices
idx = randperm(numSamples);

% Define split index (80% training, 20% testing)
splitIdx = round(0.8 * numSamples);

% Training set
X_train = X(idx(1:splitIdx), :);
y_train = y(idx(1:splitIdx), :);

% Tesing t set
X_test = X(idx(splitIdx+1:end), :);
y_test = y(idx(splitIdx+1:end), :);
% Fitting OLS model on training data
lm = fitlm(X_train, y_train);

% Displaying the  model summary
disp(lm)
% Making  predictions on test set
y_pred = predict(lm, X_test);

% Computing Mean Squared Error (MSE)
mse_test = mean((y_test - y_pred).^2);
disp(['Test Mean Squared Error: ', num2str(mse_test)])

% Computing R-squared on test set
SS_total = sum((y_test - mean(y_test)).^2);
SS_residual = sum((y_test - y_pred).^2);
R_squared_test = 1 - (SS_residual / SS_total);
disp(['Test R-squared: ', num2str(R_squared_test)])

% Scatter plot of actual vs predicted values
figure;
scatter(y_test, y_pred, 'b', 'filled');
hold on;
plot([min(y_test) max(y_test)], [min(y_test) max(y_test)], 'r--', 'LineWidth', 2);
xlabel('Actual Compressive Strength');
ylabel('Predicted Compressive Strength');
title('Actual vs Predicted Compressive Strength');
grid on;
hold off;
%% Question 2 Ridge Regression Model 
% Ridge Regression Model for data without outliers
% Loading  the cleaned dataset
Cleaned_Concrete_Data = readtable('Cleaned_Concrete_Data.csv');
summary(Cleaned_Concrete_Data);

% Defining features (X) and target variable (Y)
X = Cleaned_Concrete_Data(:, 1:8);  % First 8 columns as features
Y = Cleaned_Concrete_Data(:, 9);    % Last column as target (Compressive Strength)

% Converting tables to arrays
X = table2array(X);
Y = table2array(Y);

% Setting seed for reproducibility
rng(5);


% Getting number of samples
numSamples = size(X, 1);

% Generating random indices
idx = randperm(numSamples);

% Defining split index (80% training, 20% testing)
splitIdx = round(0.8 * numSamples);

% Training  the set
X_train = X(idx(1:splitIdx), :);
y_train = y(idx(1:splitIdx), :);

% Defining the Test set
X_test = X(idx(splitIdx+1:end), :);
y_test = y(idx(splitIdx+1:end), :);
% Fit OLS model on training data
lm = fitlm(X_train, y_train);

% Displaying the model summary
disp(lm)
% Making  predictions on test set
y_pred = predict(lm, X_test);

% Computing Mean Squared Error (MSE)
mse_test = mean((y_test - y_pred).^2);
disp(['Test Mean Squared Error: ', num2str(mse_test)])

% Computing  R-squared on test set
SS_total = sum((y_test - mean(y_test)).^2);
SS_residual = sum((y_test - y_pred).^2);
R_squared_test = 1 - (SS_residual / SS_total);
disp(['Test R-squared: ', num2str(R_squared_test)])

% Drawing the Scatter plot of actual vs predicted values
figure;
scatter(y_test, y_pred, 'b', 'filled');
hold on;
plot([min(y_test) max(y_test)], [min(y_test) max(y_test)], 'r--', 'LineWidth', 2);
xlabel('Actual Compressive Strength');
ylabel('Predicted Compressive Strength');
title('Actual vs Predicted Compressive Strength');
grid on;
hold off;

% Loading the cleaned dataset
Cleaned_Concrete_Data = readtable('Cleaned_Concrete_Data.csv');
summary(Cleaned_Concrete_Data);

% Defining the features (X) and target variable (Y)
X = Cleaned_Concrete_Data(:, 1:8);  % First 8 columns as features
Y = Cleaned_Concrete_Data(:, 9);    % Last column as target (Compressive Strength)

% Converting tables to arrays
X = table2array(X);
Y = table2array(Y);

% Setting  seed for reproducibility
rng(1);

% Splitting  data into training (80%) and test (20%) sets
numSamples = size(X, 1);
splitIdx = round(0.8 * numSamples);

% Shuffling indices for randomness
idx = randperm(numSamples);

% Assignning thr  training and test sets
X_train = X(idx(1:splitIdx), :);
Y_train = Y(idx(1:splitIdx), :);
X_test = X(idx(splitIdx+1:end), :);
Y_test = Y(idx(splitIdx+1:end), :);

% Computing mean and standard deviation from the training set
X_train_mean = mean(X_train);
X_train_std = std(X_train);

% Standardizing using training mean and std
X_train_scaled = (X_train - X_train_mean) ./ X_train_std;
X_test_scaled = (X_test - X_train_mean) ./ X_train_std;  % Use training mean/std

% Defining  lambda values for tuning
lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0];

% Initializing arrays to store Cross-Validation MSE and standard deviation
cv_mse_values = zeros(length(lambda_values), 1);
cv_std_values = zeros(length(lambda_values), 1);

% Performing 5-Fold Cross-Validation
k = 5;
cv_partition = cvpartition(size(X_train_scaled, 1), 'KFold', k);

for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    
    % Training the  Ridge Regression Model with Cross-Validation
    ridgeMdl = fitrlinear(X_train_scaled, Y_train, 'Learner', 'leastsquares', ...
                          'Regularization', 'ridge', 'Lambda', lambda, ...
                          'CrossVal', 'on', 'CVPartition', cv_partition);
    
    % Computing  MSE for each fold
    mse_values = kfoldLoss(ridgeMdl, 'Mode', 'individual');  % MSE per fold
    
    % Computing the Mean and Standard Deviation of Cross-Validation MSE
    cv_mse_values(i) = mean(mse_values);
    cv_std_values(i) = std(mse_values);
    
    % Displaying results
    fprintf('Lambda: %.4f | CV MSE: %.4f | Std: %.4f\n', lambda, cv_mse_values(i), cv_std_values(i));
end

% Finding the best lambda (lowest Cross-Validation MSE)
[~, best_lambda_idx] = min(cv_mse_values);
best_lambda = lambda_values(best_lambda_idx);
fprintf('\nBest Lambda: %.4f (Lowest Cross-Validation MSE)\n', best_lambda);

% Training the Ridge Regression using the optimal lambda
ridgeMdl_optimal = fitrlinear(X_train_scaled, Y_train, 'Learner', 'leastsquares', ...
                              'Regularization', 'ridge', 'Lambda', best_lambda);

% Predicting on test data
Y_pred_optimal = predict(ridgeMdl_optimal, X_test_scaled);

% Computing final RMSE
rmse_optimal = sqrt(mean((Y_test - Y_pred_optimal).^2));

% Computing final R-squared
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred_optimal).^2);
r_squared_optimal = 1 - (ss_residual / ss_total);

% Displaying  final results
fprintf('\nFinal Ridge Regression Model (Lambda = %.4f):\n', best_lambda);
fprintf('Final RMSE: %.4f\n', rmse_optimal);
fprintf('Final R-squared: %.4f\n', r_squared_optimal);
% Setting  seed for reproducibility
rng(1);

% Splitting  data into training (80%) and test (20%) sets
numSamples = size(X, 1);
splitIdx = round(0.8 * numSamples);

% Shuffling indices for randomness
idx = randperm(numSamples);

% Assignning thr  training and test sets
X_train = X(idx(1:splitIdx), :);
Y_train = Y(idx(1:splitIdx), :);
X_test = X(idx(splitIdx+1:end), :);
Y_test = Y(idx(splitIdx+1:end), :);

% Computing mean and standard deviation from the training set
X_train_mean = mean(X_train);
X_train_std = std(X_train);

% Standardizing using training mean and std
X_train_scaled = (X_train - X_train_mean) ./ X_train_std;
X_test_scaled = (X_test - X_train_mean) ./ X_train_std;  % Use training mean/std
% Defining  lambda values for tuning
lambda_values = [0.1, 0.5];

% Initialize arrays to store Cross-Validation MSE and standard deviation
cv_mse_values = zeros(length(lambda_values), 1);
cv_std_values = zeros(length(lambda_values), 1);

% Performing 5-Fold Cross-Validation
k = 5;
cv_partition = cvpartition(size(X, 1), 'KFold', k);

for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    
    % Training Ridge Regression Model with Cross-Validation
    ridgeMdl = fitrlinear(X_scaled, Y, 'Learner', 'leastsquares', ...
                          'Regularization', 'ridge', 'Lambda', lambda, ...
                          'CrossVal', 'on', 'CVPartition', cv_partition);
    
    % Computing MSE for each fold
    mse_values = kfoldLoss(ridgeMdl, 'Mode', 'individual');  % MSE per fold
    
    % Computing  Mean and Standard Deviation of Cross-Validation MSE
    cv_mse_values(i) = mean(mse_values);
    cv_std_values(i) = std(mse_values);
    
    % Displaying  results
    fprintf('Lambda: %.4f | CV MSE: %.4f | Std: %.4f\n', lambda, cv_mse_values(i), cv_std_values(i));
end

% Finding the  best lambda (lowest Cross-Validation MSE)
[~, best_lambda_idx] = min(cv_mse_values);
best_lambda = lambda_values(best_lambda_idx);
fprintf('\nBest Lambda: %.4f (Lowest Cross-Validation MSE)\n', best_lambda);
% Training  Ridge Regression using the optimal lambda
ridgeMdl_optimal = fitrlinear(X_train_scaled, Y_train, 'Learner', 'leastsquares', ...
                              'Regularization', 'ridge', 'Lambda', best_lambda);

% Predicting on test data
Y_pred_optimal = predict(ridgeMdl_optimal, X_test_scaled);

% Computing final RMSE
rmse_optimal = sqrt(mean((Y_test - Y_pred_optimal).^2));

% Computing final R-squared
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred_optimal).^2);
r_squared_optimal = 1 - (ss_residual / ss_total);

% Displaying the  final results
fprintf('\nFinal Ridge Regression Model (Lambda = %.4f):\n', best_lambda);
fprintf('Final RMSE: %.4f\n', rmse_optimal);
fprintf('Final R-squared: %.4f\n', r_squared_optimal);

% Defining  lambda values for tuning
lambda_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];

% Initializing the  arrays to store Cross-Validation MSE and standard deviation
cv_mse_values = zeros(length(lambda_values), 1);
cv_std_values = zeros(length(lambda_values), 1);

% Performing  5-Fold Cross-Validation
k = 5;
cv_partition = cvpartition(size(X, 1), 'KFold', k);

for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    
    % Training  Ridge Regression Model with Cross-Validation
    ridgeMdl = fitrlinear(X_scaled, Y, 'Learner', 'leastsquares', ...
                          'Regularization', 'ridge', 'Lambda', lambda, ...
                          'CrossVal', 'on', 'CVPartition', cv_partition);
    
    % Computing  MSE for each fold
    mse_values = kfoldLoss(ridgeMdl, 'Mode', 'individual');  % MSE per fold
    
    % Computing Mean and Standard Deviation of Cross-Validation MSE
    cv_mse_values(i) = mean(mse_values);
    cv_std_values(i) = std(mse_values);
    
    % Displaying the  results
    fprintf('Lambda: %.4f | CV MSE: %.4f | Std: %.4f\n', lambda, cv_mse_values(i), cv_std_values(i));
end

% Finding  optimal  lambda (lowest Cross-Validation MSE)
[~, best_lambda_idx] = min(cv_mse_values);
best_lambda = lambda_values(best_lambda_idx);
fprintf('\nBest Lambda: %.4f (Lowest Cross-Validation MSE)\n', best_lambda);
% Training Ridge Regression using the optimal lambda
ridgeMdl_optimal = fitrlinear(X_train_scaled, Y_train, 'Learner', 'leastsquares', ...
                              'Regularization', 'ridge', 'Lambda', best_lambda);

% Predicting  on test data
Y_pred_optimal = predict(ridgeMdl_optimal, X_test_scaled);

% Computing  final RMSE
rmse_optimal = sqrt(mean((Y_test - Y_pred_optimal).^2));

% Computing final R-squared
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred_optimal).^2);
r_squared_optimal = 1 - (ss_residual / ss_total);

% Displaying the  final results
fprintf('\nFinal Ridge Regression Model (Lambda = %.4f):\n', best_lambda);
fprintf('Final RMSE: %.4f\n', rmse_optimal);
fprintf('Final R-squared: %.4f\n', r_squared_optimal);
% Defining  lambda values for tuning
lambda_values = [0.01, 0.05, 0.1, 0.5, 1.0];

% Initializing the  arrays to store Cross-Validation MSE and standard deviation
cv_mse_values = zeros(length(lambda_values), 1);
cv_std_values = zeros(length(lambda_values), 1);

% Performing  5-Fold Cross-Validation
k = 5;
cv_partition = cvpartition(size(X, 1), 'KFold', k);

for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    
    % Training the  Ridge Regression Model with Cross-Validation
    ridgeMdl = fitrlinear(X_scaled, Y, 'Learner', 'leastsquares', ...
                          'Regularization', 'ridge', 'Lambda', lambda, ...
                          'CrossVal', 'on', 'CVPartition', cv_partition);
    
    % Computing  MSE for each fold
    mse_values = kfoldLoss(ridgeMdl, 'Mode', 'individual');  % MSE per fold
    
    % Computing  Mean and Standard Deviation of Cross-Validation MSE
    cv_mse_values(i) = mean(mse_values);
    cv_std_values(i) = std(mse_values);
    
    % Displaying  results
    fprintf('Lambda: %.4f | CV MSE: %.4f | Std: %.4f\n', lambda, cv_mse_values(i), cv_std_values(i));
end

% Finding the optimal  lambda (lowest Cross-Validation MSE)
[~, best_lambda_idx] = min(cv_mse_values);
best_lambda = lambda_values(best_lambda_idx);
fprintf('\nBest Lambda: %.4f (Lowest Cross-Validation MSE)\n', best_lambda);
% Train Ridge Regression using the optimal lambda
ridgeMdl_optimal = fitrlinear(X_train_scaled, Y_train, 'Learner', 'leastsquares', ...
                              'Regularization', 'ridge', 'Lambda', best_lambda);

% Predict on test data
Y_pred_optimal = predict(ridgeMdl_optimal, X_test_scaled);

% Compute final RMSE
rmse_optimal = sqrt(mean((Y_test - Y_pred_optimal).^2));

% Computing final R-squared
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred_optimal).^2);
r_squared_optimal = 1 - (ss_residual / ss_total);

% Displaying final results
fprintf('\nFinal Ridge Regression Model (Lambda = %.4f):\n', best_lambda);
fprintf('Final RMSE: %.4f\n', rmse_optimal);
fprintf('Final R-squared: %.4f\n', r_squared_optimal);
% Plot Cross-Validation MSE vs. Lambda
figure;
semilogx(lambda_values, cv_mse_values, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Lambda (log scale)');
ylabel('Cross-Validation MSE');
title('Cross-Validation MSE vs. Lambda');
grid on;

%% Question 2c-Nueral Networks
% Clearing  workspace and set seed for reproducibility
clc; clear; rng(5);

% Loading  dataset
Cleaned_Concrete_Data = readtable('Cleaned_Concrete_Data.csv');

% Extracting  features (X) and target variable (Y)
X = table2array(Cleaned_Concrete_Data(:, 1:8)); % First 8 columns as features
Y = table2array(Cleaned_Concrete_Data(:, 9));   % Last column as target

% 5-Fold Cross-Validation Setup
K = 5;
cv = cvpartition(size(X, 1), 'KFold', K);
% Standardize features (zero mean, unit variance)
X = normalize(X);
numSamples = size(X, 1);
numTrain = round(0.8 * numSamples);
idx = randperm(numSamples);
% Training the model 
X_train = X(idx(1:numTrain), :);
Y_train = Y(idx(1:numTrain), :);
X_test = X(idx(numTrain+1:end), :);
Y_test = Y(idx(numTrain+1:end), :);
%defining the layers 5-2
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(5)  % Hidden Layer 1 (5 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(5)  % Hidden Layer (2)
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('fullyConnectedLayer5-2')
%defining training options 15-2
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer 15-2');
%defining the layers 10-2
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(10)  % Hidden Layer 1 (5 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(10)  % Hidden Layer (2)
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('fullyConnectedLayer10-2')
%defining the layers 15-5
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(15)  % Hidden Layer 1 (15 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(15)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(15)  % Hidden Layer 3
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(15)  % Hidden Layer 4
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(15)  % Hidden Layer 5
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer 15-2');
%defining the layers 35-2
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(35)  % Hidden Layer 1 (35 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(35)  % Hidden Layer (35)
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('fullyConnectedLayer35-2')

%defining the layers for 25-2
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(25)  % Hidden Layer 1 (25 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(25)  % Hidden Layer (25)
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('fullyConnectedLayer25-2')
%defining the layers 30-2
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(30)  % Hidden Layer 1 (30 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(30)  % Hidden Layer (30)
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('fullyConnectedLayer30-2')
%defining the layers 50-2
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(50)  % Hidden Layer 1 (50 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer

    fullyConnectedLayer(50)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

        fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer 50-2');
%defining the layers 5-5
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(5)  % Hidden Layer 1 (5 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(5)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(5)  % Hidden Layer 3
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(5)  % Hidden Layer 4
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(5)  % Hidden Layer 5
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer5-5')
%defining the layers 15-5
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(15)  % Hidden Layer 1 (5 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(15)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(15)  % Hidden Layer 3
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(15)  % Hidden Layer 4
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(15)  % Hidden Layer 5
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer15-5')
%defining the layers 25-5
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(25)  % Hidden Layer 1 (5 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(25)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(25)  % Hidden Layer 3
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(25)  % Hidden Layer 4
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(25)  % Hidden Layer 5
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer25-5')
%defining the layers 30-5
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(30)  % Hidden Layer 1 (5 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(30)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(30)  % Hidden Layer 3
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(30)  % Hidden Layer 4
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(30)  % Hidden Layer 5
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer30-5')
%defining the layers 35-5
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(35)  % Hidden Layer 1 (35 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(35)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(35)  % Hidden Layer 3
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(35)  % Hidden Layer 4
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(35)  % Hidden Layer 5
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer35-5')
%defining the layers 50-5
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(50)  % Hidden Layer 1 (50 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(50)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(50)  % Hidden Layer 3
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(50)  % Hidden Layer 4
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(50)  % Hidden Layer 5
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer 50');
%defining the layers 60-5
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(60)  % Hidden Layer 1 (60 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(60)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(60)  % Hidden Layer 3
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(60)  % Hidden Layer 4
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

    fullyConnectedLayer(60)  % Hidden Layer 5
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer60-5')
%defining the layers 60-2
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(60)  % Hidden Layer 1 (60 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer

    fullyConnectedLayer(60)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

        fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer 60-2');
rng(5);
%% Nueral Networks for Data with Outliers
% Loading  dataset
Concrete_Data = readtable('Concrete_Data.csv');

% Extract features (X) and target variable (Y)
X = table2array(Concrete_Data(:, 1:8)); % First 8 columns as features
Y = table2array(Concrete_Data(:, 9));   % Last column as target

% 5-Fold Cross-Validation Setup
K = 5;
cv = cvpartition(size(X, 1), 'KFold', K);
% Standardize features (zero mean, unit variance)
X = normalize(X);
numSamples = size(X, 1);
numTrain = round(0.8 * numSamples);
idx = randperm(numSamples);
% Training the model 
X_train = X(idx(1:numTrain), :);
Y_train = Y(idx(1:numTrain), :);
X_test = X(idx(numTrain+1:end), :);
Y_test = Y(idx(numTrain+1:end), :);
%defining the layers 25 with outliers
layers = [
    featureInputLayer(8)  % Input layer (8 features)
    
    fullyConnectedLayer(25)  % Hidden Layer 1 (25 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer

    fullyConnectedLayer(25)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer

        fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('layer outliers25-2');
%%Question 3-pca
%%system trained PCA 
% Load dataset
Cleaned_Concrete_Data = readtable('Cleaned_Concrete_Data.csv');

% Extracting features (X) and target variable (Y)
X = table2array(Cleaned_Concrete_Data(:, 1:8)); % First 8 columns as features
Y = table2array(Cleaned_Concrete_Data(:, 9));   % Last column as target

% Standardizing features (zero mean, unit variance)
X = zscore(X);
% 5-Fold Cross-Validation Setup
K = 5;
cv = cvpartition(size(X, 1), 'KFold', K);
% Performing PCA
[coeff, score, latent, ~, explained] = pca(X);
% Performing PCA
[coeff, score, latent, ~, explained] = pca(X);

% Plotting cumulative variance explained
cumulative_variance = cumsum(explained);
% Perform PCA
[coeff, score, latent, ~, explained] = pca(X);

% Plotting cumulative variance explained
cumulative_variance = cumsum(explained);

figure;
plot(1:length(cumulative_variance), cumulative_variance, 'bo-', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained (%)');
title('PCA: Cumulative Variance Explained');
grid on;

% Finding the minimum number of PCs needed for ~95% variance
threshold = 98;  
num_PCs = find(cumulative_variance >= threshold, 1);

fprintf('Number of Principal Components selected: %d\n', num_PCs);


% Choosing number of principal components based on variance explained
num_PCs = 7; 
% Reducing dimensionality
X_PCA = score(:, 1:num_PCs);  % Use first `num_PCs` components

% Splitting data (80% Train, 20% Test)
numSamples = size(X_PCA, 1);
numTrain = round(0.8 * numSamples);
idx = randperm(numSamples);

X_train = X_PCA(idx(1:numTrain), :);
Y_train = Y(idx(1:numTrain), :);
X_test = X_PCA(idx(numTrain+1:end), :);
Y_test = Y(idx(numTrain+1:end), :);
layers = [
    featureInputLayer(7)  % Input layer (8 features)
    
    fullyConnectedLayer(25)  % Hidden Layer 1 (50 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(25)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);
% Displaying the  Results
fprintf('Neural Network Regression using PCA:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('for 98');
% Finding  the minimum number of PCs needed for ~90% variance
% Find the minimum number of PCs needed for ~95% variance
threshold = 95;  
num_PCs = find(cumulative_variance >= threshold, 1);

fprintf('Number of Principal Components selected: %d\n', num_PCs);


% Choosing  number of principal components based on variance explained
num_PCs = 6; 
% Reduce dimensionality
X_PCA = score(:, 1:num_PCs);  % Use first `num_PCs` components

% Splitting the  data (80% Train, 20% Test)
numSamples = size(X_PCA, 1);
numTrain = round(0.8 * numSamples);
idx = randperm(numSamples);

X_train = X_PCA(idx(1:numTrain), :);
Y_train = Y(idx(1:numTrain), :);
X_test = X_PCA(idx(numTrain+1:end), :);
Y_test = Y(idx(numTrain+1:end), :);
layers = [
    featureInputLayer(6) % Input layer (6 features)
    
    fullyConnectedLayer(25)  % Hidden Layer 1 (50 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(25)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);
% Display Results
fprintf('Neural Network Regression using PCA:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('for 95');
% Finding  the minimum number of PCs needed for ~90% variance
threshold = 90;  
num_PCs = find(cumulative_variance >= threshold, 1);

fprintf('Number of Principal Components selected: %d\n', num_PCs);


% Choosing the  number of principal components based on variance explained
num_PCs = 6; 
% Reducing the  dimensionality
X_PCA = score(:, 1:num_PCs);  

% Spliting data (80% Train, 20% Test)
numSamples = size(X_PCA, 1);
numTrain = round(0.8 * numSamples);
idx = randperm(numSamples);

X_train = X_PCA(idx(1:numTrain), :);
Y_train = Y(idx(1:numTrain), :);
X_test = X_PCA(idx(numTrain+1:end), :);
Y_test = Y(idx(numTrain+1:end), :);

% Finding the minimum number of PCs needed for ~95% variance
threshold = 90;  
num_PCs = find(cumulative_variance >= threshold, 1);

fprintf('Number of Principal Components selected: %d\n', num_PCs);


% Choosing the  number of principal components based on variance explained
num_PCs = 6; 
% Reducing the dimensionality
X_PCA = score(:, 1:num_PCs);  % Use first `num_PCs` components

% Splitting the  data (80% Train, 20% Test)
numSamples = size(X_PCA, 1);
numTrain = round(0.8 * numSamples);
idx = randperm(numSamples);

X_train = X_PCA(idx(1:numTrain), :);
Y_train = Y(idx(1:numTrain), :);
X_test = X_PCA(idx(numTrain+1:end), :);
Y_test = Y(idx(numTrain+1:end), :);
layers = [
    featureInputLayer(7)  % Input layer (8 features)
    
    fullyConnectedLayer(25)  % Hidden Layer 1 (50 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(25)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Display Results
fprintf('Neural Network Regression using PCA:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('for 90');
% Finding  the minimum number of PCs needed for ~99% variance
threshold = 99;  
num_PCs = find(cumulative_variance >= threshold, 1);

fprintf('Number of Principal Components selected: %d\n', num_PCs);


% Choosing the  number of principal components based on variance explained
num_PCs = 7; 
% Reduce dimensionality
X_PCA = score(:, 1:num_PCs);  

% Spliting data (80% Train, 20% Test)
numSamples = size(X_PCA, 1);
numTrain = round(0.8 * numSamples);
idx = randperm(numSamples);

X_train = X_PCA(idx(1:numTrain), :);
Y_train = Y(idx(1:numTrain), :);
X_test = X_PCA(idx(numTrain+1:end), :);
Y_test = Y(idx(numTrain+1:end), :);

% Defining Neural Network Architecture (2 layers, 25neurons per layer)
% Find the minimum number of PCs needed for ~95% variance
threshold = 98;  
num_PCs = find(cumulative_variance >= threshold, 1);

fprintf('Number of Principal Components selected: %d\n', num_PCs);


% Choose number of principal components based on variance explained
num_PCs = 7; 
% Reduce dimensionality
X_PCA = score(:, 1:num_PCs);  % Use first `num_PCs` components

% Split data (80% Train, 20% Test)
numSamples = size(X_PCA, 1);
numTrain = round(0.8 * numSamples);
idx = randperm(numSamples);

X_train = X_PCA(idx(1:numTrain), :);
Y_train = Y(idx(1:numTrain), :);
X_test = X_PCA(idx(numTrain+1:end), :);
Y_test = Y(idx(numTrain+1:end), :);
layers = [
    featureInputLayer(7)  % Input layer (8 features)
    
    fullyConnectedLayer(25)  % Hidden Layer 1 (50 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(25)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);
% Displaying Results
fprintf('Neural Network Regression using PCA:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('for 99');
%% Water-Bind Ratio PCA 
% Loading dataset
data = readtable('Cleaned_Concrete_Data.csv');

% Extracting the required features
CoarseAggregate = data.CoarseAggregate;
FineAggregate = data.FineAggregate;
Age = data.Age;

% Computing the  Water-Binder Ratio
Binder = data.Cement + data.BlastFurnaceSlag + data.FlyAsh + data.Superplasticizer;
WaterBinderRatio = data.Water ./ Binder;

% Creating reduced feature set as a table
X_reduced = table(WaterBinderRatio, CoarseAggregate, FineAggregate, Age);

% Extracting target variable
Y = table(data.Strength, 'VariableNames', {'Strength'});

% Combining into a new table
ReducedData = [X_reduced, Y];

% Displaying the new table in MATLAB
disp(ReducedData);

% Saving the new dataset to a CSV file
writetable(ReducedData, 'Reduced_Concrete_Data.csv');

% Confirming file creation
disp('Reduced dataset saved as Reduced_Concrete_Data.csv');

% Loading the reduced dataset
Reduced_Concrete_Data = readtable('Reduced_Concrete_Data.csv');
% Display new data in the program
disp(Reduced_Concrete_Data);
X = table2array(Reduced_Concrete_Data(:, 1:4)); % First 4 columns as features
Y = table2array(Reduced_Concrete_Data(:, 5));   % Last column as target

% 5-Fold Cross-Validation Setup
K = 5;
cv = cvpartition(size(X, 1), 'KFold', K);
% Standardizing features (zero mean, unit variance)
X = normalize(X);
numSamples = size(X, 1);
numTrain = round(0.8 * numSamples);
idx = randperm(numSamples);
% Training the model 
X_train = X(idx(1:numTrain), :);
Y_train = Y(idx(1:numTrain), :);
X_test = X(idx(numTrain+1:end), :);
Y_test = Y(idx(numTrain+1:end), :);
%defining the layers
layers = [
    featureInputLayer(4)  % Input layer (4features)
    
    fullyConnectedLayer(25)  % Hidden Layer 1 (25 neurons)
    batchNormalizationLayer
    dropoutLayer(0.2) 
    reluLayer
    
    fullyConnectedLayer(25)  % Hidden Layer 2
    batchNormalizationLayer
    dropoutLayer(0.2)
    reluLayer  
    fullyConnectedLayer(1)  % Output layer (1 neuron for regression)
    regressionLayer];
%defining training options 
options = trainingOptions('adam', ...  
    'InitialLearnRate', 0.001, ...     
    'MaxEpochs', 100, ...              
    'MiniBatchSize', 32, ...           
    'Shuffle', 'every-epoch', ...     
    'Plots', 'training-progress', ...  
    'Verbose', false);
%training the model
net = trainNetwork(X_train, Y_train, layers, options);
%computing rmse
Y_pred = predict(net, X_test);  % Get predictions from the trained model
rmse = sqrt(mean((Y_test - Y_pred).^2));
% Computing R-squared (R²)
ss_total = sum((Y_test - mean(Y_test)).^2);
ss_residual = sum((Y_test - Y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% Displaying  Results
fprintf('4-layer Neural Network Regression Results:\n');
fprintf('RMSE: %.4f\n', rmse);
fprintf('R-squared: %.4f\n', r_squared);
fprintf('Reduced');