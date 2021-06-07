clc;
clear all;
close all;
% A = zeros(15,542);
% E = zeros(14,542);
V = zeros(15,542);

for i=1:15
 load(sprintf('vehicle%d.mat',i));
end

V = [[V1];[V2];[V3];[V4];[V5];[V6];[V7];[V8];[V9];[V10];[V11];[V12];[V13];[V14];[V15]];%[V16];[V17];[V18];[V19];[V20];[V21];[V22];[V23];[V24];[V25]];
% A = [[A1];[A2];[A3];[A4];[A5]];
% E = [[E2];[E3];[E4];[E5];[E6]];

var=mean(max(V)-min(V))


