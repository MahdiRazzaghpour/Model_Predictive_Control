clc;
clear all;
close all;
A = zeros(15,485);
E = zeros(14,485);
V = zeros(15,485);

for i=1:15
 load(sprintf('vehicle%d.mat',i));
end

V = [[V1];[V2];[V3];[V4];[V5];[V6];[V7];[V8];[V9];[V10];[V11];[V12];[V13];[V14];[V15]];
A = [[A1];[A2];[A3];[A4];[A5];[A6];[A7];[A8];[A9];[A10];[A11];[A12];[A13];[A14];[A15]];
E = [[E2];[E3];[E4];[E5];[E6];[E7];[E8];[E9];[E10];[E11];[E12];[E13];[E14];[E15]];

figure(1)
subplot(3,1,1);
plot(flip(V',1))
xlabel('Time [ms]','FontSize',20);
ylabel('Speed [m/s]','FontSize',20);
lgd=legend('Vehicle 0','Vehicle 1','Vehicle 2','Vehicle 3','Vehicle 4','Vehicle 5','Vehicle 6','Vehicle 7','Vehicle 8','Vehicle 9','Vehicle 10','Vehicle 11','Vehicle 12','Vehicle 13','Vehicle 14');
subplot(3,1,2);
plot(flip(A',1))
xlabel('Time [ms]','FontSize',20);
ylabel('Acceleration [m/S^2]','FontSize',20);
subplot(3,1,3); 
plot(flip(E',1))
xlabel('Time [ms]','FontSize',20);
ylabel('Spacing Error [s]','FontSize',20);



% plot(PER,five,'r-o','LineWidth',4);
% hold on 
% plot(PER,ten,'b-o','LineWidth',4);
% xlabel('PER','FontSize',24);
% ylabel('95% Error [m]','FontSize',24);
% legend('Five Vehicle','Ten Vehicle','FontSize',24);
% grid on;
% grid minor;