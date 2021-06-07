clc;
close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('mah.r.888.CC.mat');
file = table2array(RangeMat);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = file(file(:,4)<=100,:);      % identifying the communication radius
%file = file(file(:,4)>=300,:);

ego_id = unique(file(:,2));         % unique recivers

counter = 1;
total = 1;
for i = 1:size(ego_id,1)
    recived_i = file(file(:,2)== ego_id(i,1),:);  %recived packets for car i
    remote_id = unique(recived_i(:,3));           % unique transmiters for car i
    
    for j = 1:size(remote_id,1)
        recived_j = recived_i(recived_i(:,3)== remote_id(j,1),:);  %packets between specific reciver and transmiter
        recived_j = sort(recived_j,8); % Sorted packets based on transmision time

        for k=1:size(recived_j,1)-1
            X = recived_j(k+1,1) - recived_j(k,1);  % diffrence between consecutive packets
            ipg(counter,k) = round(double(X),-2);   % Rounding them to nearest 100
            ipg_total(1,total) = round(double(X),-2);
            total = total+1;
        end
        counter = counter+1;
    end
end

[f,x] = ecdf(ipg_total(ipg_total>0));  %CDF
figure(1)
plot(x,f)

for i = 1:size(f,1)-1
    pdf(i,1) = f(i+1,1) - f(i,1);   %calculation the pdf
end

figure(2)
stem(x(2:end),pdf);

ipg_value = unique(ipg_total(ipg_total>0));

for k=1:size(ipg_value,2)
    N=1;  %counter
    for i=1:size(ipg,1)
        for j=1:size(ipg,2)-1
            if (ipg(i,j)== ipg_value(1,k)) && (ipg(i,j+1)~= 0)
                next_ipg(k,N) = ipg(i,j+1);
                N=N+1;
            end
        end
    end
end

for k=1:size(next_ipg,1)
    temp1 = next_ipg(k,:);
    temp = temp1(temp1>0);
    if isempty(temp) ~= 1
        [f1,x1] = ecdf(temp);
            for i = 1:size(f1,1)-1
                Transition_Probability(i,k) = f1(i+1,1) - f1(i,1);  %calculating the transition probabilities
            end
    end
end

MC_Transition = Transition_Probability(1:10,1:10);
MC_Transition = MC_Transition ./ sum(MC_Transition,1); %Probabilty normalization
mc = dtmc(MC_Transition');

figure(3);
graphplot(mc,'ColorEdges',true);

mc_ipg = simulate(mc,62000)*100;

[mc_f,mc_x] = ecdf(mc_ipg);  %CDF
figure(4)
plot(mc_x,mc_f)
hold on
plot(x,f)

for i = 1:size(mc_f,1)-1
    mc_pdf(i,1) = mc_f(i+1,1) - mc_f(i,1);   %calculation the pdf
end

figure(5)
stem(mc_x(2:end),mc_pdf);
hold on
stem(x(2:end),pdf);
