close all
clc

raw_data = load('Data39.csv');

P_Q = raw_data(:,3:4);
V_T = raw_data(:,1:2);

clear raw_data;

[dim_c, dim_r] = size(P_Q);

P_Q_Fit = zeros([dim_c dim_r]);

for i = 1:2
    P_Q_Fit(:,i)=(P_Q(:,i)-min(P_Q(:,i)))/(max(P_Q(:,i))-min(P_Q(:,i)));
end

P_Q_TP = P_Q_Fit';
V_T_TP = V_T';

hidden_layer_neuron = 5;
 
net = fitnet(hidden_layer_neuron);
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100;

start_train_time = cputime;

[net, tr] = train(net, P_Q_TP, V_T_TP);

end_train_time = cputime;

Y_VnT = net(P_Q_TP)';
Y_VnT_Train = exp(net(P_Q_TP(:,tr.trainInd))) - 1;
Y_VnT_TrainTrue = exp(V_T_TP(:, tr.trainInd)) - 1;

rmse_V = sqrt(mean((Y_VnT_Train(1,:) - Y_VnT_TrainTrue(1,:)).^2));
rmse_T = sqrt(mean((Y_VnT_Train(2,:) - Y_VnT_TrainTrue(2,:)).^2));

[acc_V, acc_T] = accuracy_score(V_T, Y_VnT);

total_time = end_train_time - start_train_time;

fprintf("\nAkurasi Output Tegangan   : %.2f %c", acc_V * 100, '%');
fprintf("\nAkurasi Output Sudut Fasa : %.2f %c", acc_T * 100, '%');
fprintf("\nWaktu Training            : %.2f s\n", total_time);


fprintf("\nV Aktual\tSudut Fasa Aktual\tV Prediksi\tSudut Fasa Prediksi\n");
fprintf("========\t=================\t==========\t===================\n");

for i=1:size(P_Q, 1)
    fprintf("%.4f\t\t%.4f   \t\t%.4f\t\t%.4f\n", V_T(i,1), V_T(i,2), Y_VnT(i,1), Y_VnT(i,2));
end

function [v, t] = accuracy_score(y, z)
    V_actual_total = round(y(:,1));
    T_actual_total = round(y(:,2));
    V_pred_total = round(z(:,1));
    T_pred_total = round(z(:,2));
    disp(length(V_actual_total));
    v = mean(double(V_actual_total == V_pred_total));
    t = mean(double(T_actual_total == T_pred_total));
end

