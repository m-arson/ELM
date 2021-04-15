close all
clc

raw_data = load("Data12.csv");

% Training Data

P_Q = raw_data(1:9,3:4);
V_T = raw_data(1:9,1:2);

% Testing Data

P_Q_test = raw_data(10:end, 3:4);
V_T_test = raw_data(10:end, 1:2);

clear raw_data;

P_Q_size = size(P_Q, 2);
hidden_layer_neurons = 5;

W = zeros(hidden_layer_neurons, P_Q_size);
r_min = -0.5;
r_max = 0.5;

for i=1:P_Q_size
    cnt = 0;
    for j=1:hidden_layer_neurons
       r1 = randi([-1, 1]);
       r2 = rand(1,1, 'double');
       r3 = r1 * r2;
       if r3 > 0.5 || r3 < 0.5
           r3 = r3/2;
       end
       if(r3 == 0)
           if(cnt >= 1)
               r3 = rand(1,1,'double') / 2;
           else
               cnt = cnt + 1;
           end
       end
       W(j,i) =  r3;
    end
end

start_training_time = cputime;

H_init = P_Q * W';
H = 1 ./ (1 + exp(-H_init));
H_plus = (H' * H)^-1 * H';
beta = H_plus * V_T;
Y_VnT = H * beta;

end_training_time = cputime;

MAPE = abs(mean(remove_zero(abs(Y_VnT - V_T) / V_T))) * 100;

fprintf("\nTRAINING\n\n");
fprintf("MAPE                      : %.5f\n", MAPE);
fprintf("Waktu Training            : %.5f s\n", end_training_time - start_training_time);

% Testing 
start_testing_time = cputime;

H_init_test = P_Q_test * W';
H_test = 1 ./ (1 + exp(-H_init_test));
Y_VnT_test = H_test * beta;

end_testing_time = cputime;

[acc_V, acc_T] = accuracy_score(V_T, Y_VnT, V_T_test, Y_VnT_test);

fprintf("\nTESTING\n");
fprintf("\nAkurasi Testing Tegangan   : %.2f %c", acc_V * 100, '%');
fprintf("\nAkurasi Testing Sudut Fasa : %.2f %c", acc_T * 100, '%');
fprintf("\nWaktu Testing              : %.5f s\n", end_testing_time - start_testing_time);

fprintf("\nV Aktual\tSudut Fasa Aktual\tV Prediksi\tSudut Fasa Prediksi\n");
fprintf("========\t=================\t==========\t===================\n");

for i=1:size(P_Q, 1)
    fprintf("%.4f\t\t%.4f   \t\t%.4f\t\t%.4f\n", V_T(i,1), V_T(i,2), Y_VnT(i,1), Y_VnT(i,2));
end
for i=1:size(P_Q_test, 1)
    fprintf("%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", V_T_test(i, 1), V_T_test(i, 2), Y_VnT_test(i, 1), Y_VnT_test(i, 2));
end

function x = remove_zero(y)
    ind = (y == 0);
    y(ind) = [];
    x = y;
end

function [v, t] = accuracy_score(w, x, y, z)
    V_actual_total = [round(w(:,1));round(y(:,1))];
    T_actual_total = [round(w(:,2));round(y(:,2))];
    V_pred_total = [round(x(:,1));round(z(:,1))];
    T_pred_total = [round(x(:,2));round(z(:,2))];
    v = mean(double(V_actual_total == V_pred_total));
    t = mean(double(T_actual_total == T_pred_total));
end