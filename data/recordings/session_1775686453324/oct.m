% Read CSV, skipping the header
data = csvread('accel.csv', 1, 0);

% Separate columns
timestamps = data(:, 1);
x = data(:, 2);
y = data(:, 3);
z = data(:, 4);

figure;
plot(timestamps, x, 'r'); hold on;
plot(timestamps, y, 'g');
plot(timestamps, z, 'b');
xlabel('Time (s)');
ylabel('Acceleration (g)');
title('Smartwatch Accelerometer Data');
legend('X','Y','Z');
grid on;
