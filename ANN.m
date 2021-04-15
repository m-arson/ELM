close all
clc

data = dlmread('/home/arson/Documents/Data12.csv');

x = data(:,3:4);
y = data(:,1:2);

clear data;

[dim_c, dim_r] = size(x);

x2 = zeros([dim_c dim_r]);

for i = 1:2
    x2(:,i)=(x(:,i)-min(x(:,i)))/(max(x(:,i))-min(x(:,i)));
end

xt = x';
yt = y';
%y2t = y(:,2)';
 
hiddenLayerSize = 5;
 
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100;

start_time = cputime;

[net, tr] = train(net, xt, yt);

end_time = cputime;
total_time = end_time - start_time;

yTrain = exp(net(xt(:,tr.trainInd))) - 1;
yTrainTrue = exp(yt(:, tr.trainInd)) - 1;

rmse_y1 = sqrt(mean((yTrain(1,:) - yTrainTrue(1,:)).^2));
rmse_y2 = sqrt(mean((yTrain(2,:) - yTrainTrue(2,:)).^2));
disp(rmse_y2);
disp(rmse_y1);

fprintf("%f s\n", total_time);
fprintf("%f %c\n", rmse_y1, '%');
fprintf("%f %c\n", rmse_y2, '%');

val = net(xt)';
