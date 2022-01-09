clear;
X=[4.6019, 5.2564, 5.2200, 3.2886, 3.7942
3.2271, 4.9275, 3.2789, 5.7019, 3.9945
3.8936, 6.7906, 7.1624, 4.1807, 4.9630 
6.9630, 4.4597, 6.7175, 5.8198, 5.0555 
4.6469, 6.6931, 5.7111, 4.3672, 5.3927 
4.1220, 5.1489, 6.5319, 5.5318, 4.2403 
5.3480, 4.3022, 7.0193, 3.2063, 4.3405 
5.7715, 4.1797, 5.0179, 5.6545, 6.2577 
4.0729, 4.8301, 4.5283, 4.8858, 5.3695
4.3814, 5.8001, 5.4267, 4.5277, 5.2760];
x = -50:.1:50;
hn = [0.1 0.5 1 2 5 10];
%xIntegrate = 
%% 以下讨论方窗的情况
%{
sum_more_general = zeros(length(x),length(hn));
for xi = 1:length(x)
    for hni = 1:length(hn)
        for Xi = 1:length(X)
            if abs(x(xi)-X(Xi)) <= hn(hni)/2
                sum_more_general(xi,hni) = sum_more_general(xi,hni) + 1/hn(hni);
            else
                sum_more_general(xi,hni) = sum_more_general(xi,hni);
            end
        end
    end
end
Pn = sum_more_general / length(X);
% 以下作图
for i = 1 : length(hn)
    figure(1);
    subplot(2,3,i);
    plot(x,Pn(:,i));xlabel('x');ylabel('Pn(x)');title(['方窗下hn=',num2str(hn(i)),'的概率密度函数']);
    plot(x,Pn(:,i));xlabel('x');ylabel('Pn(x)');title(['正态窗下hn=',num2str(hn(i)),'的概率密度函数']);
    if i == 1
        axis([1,10,0,3]);
    elseif i == 2 || i == 3 || i == 4
        axis([1,10,0,0.8]);
    else
        axis([1,10,0,0.3]);
    end
end
%}
%% 以下讨论正态窗的情况
sum_normal = zeros(length(x),length(hn));
sum_all = zeros(1,length(hn));
pi = 3.14;
cons = 1/sqrt(2)*1/pi;
for xi = 1:length(x)
    for hni = 1:length(hn)
        for Xi = 1:length(X)
                tmpx = -1/2*(X(Xi)-x(xi))*(X(Xi)-x(xi))/(hn(hni)*hn(hni));
                sum_normal(xi,hni) = sum_normal(xi,hni) + 1/hn(hni)*cons*exp(tmpx);
        end
    end
end
sum_normal_tmp = sum_normal;
for hnj = 1:length(hn)
    sum_all(1,hnj) = sum(sum_normal(:,hnj));
    for xj = 1: length(x)
        sum_normal(xj,hnj) = sum_normal_tmp(xj,hnj)/sum_all(1,hnj);
    end
end
%Pn = sum_normal / length(X);
Pn = sum_normal;
% 以下作图
for i = 1 : length(hn)
    figure(1);
    subplot(2,3,i);
    plot(x,Pn(:,i));xlabel('x');ylabel('Pn(x)');title(['正态窗下hn=',num2str(hn(i)),'的概率密度函数']);
    if i == 1
        axis([-5,10,0,0.15]);
    elseif i == 2 || i == 3 || i == 4
        axis([-5,15,0,0.06]);
    elseif i ==5
        axis([-10,20,0,0.02]);
    else 
        axis([-35,25,0,0.02]);
    end
end