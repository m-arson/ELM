close all
clc

raw_data = load("Data12.csv");

% Training Data

P_Q = raw_data(1:9,3:4);
V = raw_data(1:9,1);
theta = raw_data(1:9,2);

% Testing Data

P_Q_test = raw_data(10:end, 3:4);
V_test = raw_data(10:end, 1);
theta_test = raw_data(10:end, 2);

clear raw_data;

P_Q_size = size(P_Q, 2);
hidden_layer = 5;

W = zeros(hidden_layer, P_Q_size);
r_min = -0.5;
r_max = 0.5;

for i=1:P_Q_size
    cnt = 0;
    for j=1:hidden_layer
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

beta_V = H_plus * V;
beta_theta = H_plus * theta;

Y_V = H * beta_V;
Y_theta = H * beta_theta;

end_training_time = cputime;

MAPE_V = abs(mean(remove_zero(abs(Y_V - V) / V))) * 100.00;
MAPE_theta = abs(mean(remove_zero(abs(Y_theta - theta) / theta))) * 100.00;

fprintf("\nTRAINING\n\n");

fprintf("MAPE Tegangan Magnitudo : %.2f\n", MAPE_V);
fprintf("MAPE Sudut Fasa         : %.2f\n", MAPE_theta);
fprintf("Waktu Training          : %.2f s\n", end_training_time - start_training_time);

% Testing 
start_testing_time = cputime;

H_init_test = P_Q_test * W';
H_test = 1 ./ (1 + exp(-H_init_test));
Y_V_test = H_test * beta_V;
Y_theta_test = H_test * beta_theta;

end_testing_time = cputime;

acc_V =  100 - abs(abs(mean(Y_V_test)) - abs(mean(V_test))) / abs(mean(V_test));
acc_theta =  100 - abs(abs(mean(Y_theta_test)) - abs(mean(theta_test))) / abs(mean(theta_test));

fprintf("\nTESTING\n");

fprintf("\nAkurasi Tes Tegangan    : %.2f %c", acc_V, '%');
fprintf("\nAkurasi Tes Sudut Fasa  : %.2f %c", acc_theta, '%');
fprintf("\nWaktu Testing           : %.2f s\n", end_testing_time - start_testing_time);

fprintf("\nV Aktual\tSudut Fasa Aktual\tV Prediksi\tSudut Fasa Prediksi\n");
fprintf("========\t=================\t==========\t===================\n");

for i=1:size(P_Q, 1)
    fprintf("%.4f\t\t%.4f   \t\t%.4f\t\t%.4f\n", V(i), theta(i), Y_V(i), Y_theta(i));
end
for i=1:size(P_Q_test, 1)
    fprintf("%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", V_test(i), theta_test(i), Y_V_test(i), Y_theta_test(i));
end

function x = remove_zero(y)
    ind = (y == 0);
    y(ind) = [];
    x = y;
end
