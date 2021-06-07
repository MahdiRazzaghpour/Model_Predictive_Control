clc;
clear all;
close all;
A = zeros(15,620);
E = zeros(14,620);
V = zeros(15,620);

for i=1:15
 load(sprintf('vehicle%d.mat',i));
end

V = [[V1];[V2];[V3];[V4];[V5];[V6];[V7];[V8];[V9];[V10];[V11];[V12];[V13];[V14];[V15]];
A = [[A1];[A2];[A3];[A4];[A5];[A6];[A7];[A8];[A9];[A10];[A11];[A12];[A13];[A14];[A15]];
E = [[E2];[E3];[E4];[E5];[E6];[E7];[E8];[E9];[E10];[E11];[E12];[E13];[E14];[E15]];

N=1;
perc=zeros(1,8680);
for i=1:14
    for j = 1:size(E,2)
        perc(1,N) = E(i,j);
        N=N+1;
    end
end

Y = prctile(perc,95)