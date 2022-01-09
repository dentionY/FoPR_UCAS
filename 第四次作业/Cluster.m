sample = [0.697	0.460;0.774	0.376;0.634	0.264;0.608	0.318;0.556	0.215;
          0.403 0.237;0.481	0.149;0.437	0.211;0.666	0.091;0.243	0.267;
          0.245 0.057;0.343 0.099;0.639 0.161;0.657 0.198;0.360 0.370;
          0.593	0.042;0.719	0.103;0.359	0.188;0.339	0.241;0.282	0.257;
          0.748	0.232;0.714	0.346;0.483	0.312;0.478	0.437;0.525	0.369;
          0.751	0.489;0.532	0.472;0.473	0.376;0.725	0.445;0.446	0.459];
%% K-均值聚类
[number,dimension] = size(sample);
K = 4;
%center_cluster = zeros(K, dimension);
% 初始化center_cluster
tmp_index = randperm(30);
initial_index = tmp_index([1,8,15,22]);
center_cluster = sample(initial_index,:);
Je = 100;
class_sample = zeros(30,1);
Je_vex = [];
iteration = 0;
center_dist = 100;
center_dist_vec = [];

while center_dist > 0.0001
    iteration = iteration + 1;
    % 将所有样本分为4类，依次被标记为1，2，3，4  并更新Je
    Je_tmp = 0;
    %center_dist_tmp = 0;
    center_cluster_tmp = zeros(4,2);
    for numberi = 1:number
        dist_four_center = zeros(1,4);
        for Ki = 1:K
            dist_tmp1 = (center_cluster(Ki,1)-sample(numberi,1))*(center_cluster(Ki,1)-sample(numberi,1))+(center_cluster(Ki,2)-sample(numberi,2))*(center_cluster(Ki,2)-sample(numberi,2));
            dist_tmp2 = sqrt(dist_tmp1);
            dist_four_center(1,Ki) = dist_tmp2;
        end
        [dist_four_center_seq,seq_index] = sort(dist_four_center);
        class_sample(numberi,1) = seq_index(1);
        Je_tmp = Je_tmp + dist_four_center(1,seq_index(1));
    end
    Je = Je_tmp;
    Je_vex = [Je_vex Je];
    % 更新分类中心
    class1 = find(class_sample==1);center_cluster_tmp(1,:) = mean(sample(class1,:));
    class2 = find(class_sample==2);center_cluster_tmp(2,:) = mean(sample(class2,:));
    class3 = find(class_sample==3);center_cluster_tmp(3,:) = mean(sample(class3,:));
    class4 = find(class_sample==4);center_cluster_tmp(4,:) = mean(sample(class4,:));
    center_dist = sum([norm(center_cluster(1,:)-center_cluster_tmp(1,:)),norm(center_cluster(2,:)-center_cluster_tmp(2,:)),norm(center_cluster(3,:)-center_cluster_tmp(3,:)),norm(center_cluster(4,:)-center_cluster_tmp(4,:))]);
    center_dist_vec = [center_dist_vec center_dist];
    center_cluster = center_cluster_tmp;
end
subplot(1,2,1)
plot(sample(class1,1),sample(class1,2),'g+',sample(class2,1),sample(class2,2),'r+',sample(class3,1),sample(class3,2),'b+',sample(class4,1),sample(class4,2),'m+');
hold on
plot(center_cluster(1,1),center_cluster(1,2),'g*',center_cluster(2,1),center_cluster(2,2),'r*',center_cluster(3,1),center_cluster(3,2),'b*',center_cluster(4,1),center_cluster(4,2),'m*');
title('各点分类情况');
subplot(1,2,2)
plot(1:iteration,center_dist_vec,'k-');
set(gca,'XTick',1:1:iteration);
title('聚类中心变化总和随训练次数变化');