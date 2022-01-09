sample = [0.697	0.460;0.774	0.376;0.634	0.264;0.608	0.318;0.556	0.215;
          0.403 0.237;0.481	0.149;0.437	0.211;0.666	0.091;0.243	0.267;
          0.245 0.057;0.343 0.099;0.639 0.161;0.657 0.198;0.360 0.370;
          0.593	0.042;0.719	0.103;0.359	0.188;0.339	0.241;0.282	0.257;
          0.748	0.232;0.714	0.346;0.483	0.312;0.478	0.437;0.525	0.369;
          0.751	0.489;0.532	0.472;0.473	0.376;0.725	0.445;0.446	0.459];
%% 模糊K-均值聚类  考虑b=2
[number,dimension] = size(sample);
K = 4;
%center_cluster = zeros(K, dimension);
% 初始化center_cluster
tmp_index = randperm(30);
initial_index = tmp_index([1,8,15,22]);
center_cluster = sample(initial_index,:);
delta_cluster = 100;%------------------------------------------------------聚类中心变化值（采用2范数）
iteration = 0;%------------------------------------------------------------迭代次数
delta_cluster_vec = [];%---------------------------------------------------存储聚类中心2范数变化值的向量，用于记录直到循环结束
initial_degree = zeros(30,4); %--------------------------------------------初始隶属度的定义
degree = zeros(30,4);%-----------------------------------------------------隶属度的定义

while delta_cluster > 0.00001
    iteration = iteration + 1;
    %% 计算隶属度
    %Je_tmp = 0;
    for numberi = 1:number
        tmp_degree = zeros(30,4);%----------------------------------------隶属度赋值的中间量
        % 以下4个signal矢量(均为1×2)是为了判断sample与聚类中心重合的情况，如果不排除将会在计算隶属度时出现分母为0
        % signal的计算方式是每个sample和4个聚类中心坐标相减
        signal1 = (sample(numberi,:)-center_cluster(1,:));%---------------此处是为了排除sample和聚类中心重合而导致无法对隶属度赋值的干扰
        signal2 = (sample(numberi,:)-center_cluster(2,:));
        signal3 = (sample(numberi,:)-center_cluster(3,:));
        signal4 = (sample(numberi,:)-center_cluster(4,:));
         if or(signal1(1),signal1(2)) && or(signal2(1),signal2(2)) && or(signal3(1),signal3(2)) && or(signal4(1),signal4(2))
             for Ki = 1:K
                 degree_tmp1 = (center_cluster(Ki,1)-sample(numberi,1))*(center_cluster(Ki,1)-sample(numberi,1))+(center_cluster(Ki,2)-sample(numberi,2))*(center_cluster(Ki,2)-sample(numberi,2));
                 degree_tmp2 = 1/degree_tmp1;
                 tmp_degree(numberi,Ki) = degree_tmp2;%--------------------sample和聚类中心不重合的情况下计算隶属度
             end
         % 以下else考虑sample和某个聚类中心重合，那么需要寻找和哪个聚类中心重合，以便于让该sample以1的隶属度属于当前聚类中心，以0的隶属度属于其他聚类中心
         else
             tmp_degree(numberi,:) = 0;
             tmp_degree_index1 = find(sample(numberi,1) == center_cluster(:,1));
             tmp_degree_index2 = find(sample(numberi,2) == center_cluster(:,2));
             %if tmp_degree_index1 == tmp_degree_index2
             tmp_degree_index12 =intersect(tmp_degree_index1,tmp_degree_index2); 
             tmp_degree(numberi,tmp_degree_index12) = 1;
        end
        for Ki = 1:K
            degree(numberi,Ki) = tmp_degree(numberi,Ki)/sum(tmp_degree(numberi,:));%-----------------------隶属度中间变量对隶属度的赋值
        end
    end
    % 存储初始隶属度，这是题目要求
    if iteration == 1
        initial_degree = degree;%-------------------------------------------初始隶属度的赋值
    end
    %% 更新聚类中心
    % 以下3行是为了提前准备聚类中心更新公式的隶属度平方，其中
    % power_two_degree是隶属度的平方，采用点积的形式
    % power_sum是power_two_degree每一列的总和，也就是1×4的向量
    % tmp_center_cluster是聚类中心赋值的中间变量，因为考虑到需要计算聚类中心2范数的变化值。
    power_two_degree = degree.*degree;   %b=2  
    power_sum = [sum(power_two_degree(:,1)) sum(power_two_degree(:,2)) sum(power_two_degree(:,3)) sum(power_two_degree(:,4))];
    tmp_center_cluster = zeros(K,2);
    for i = 1:K
        tmp = [power_two_degree(:,i) power_two_degree(:,i)].*sample;
        tmp_center_cluster(i,1) = sum(tmp(:,1))/power_sum(i);
        tmp_center_cluster(i,2) = sum(tmp(:,2))/power_sum(i);
    end
    %更新delta_cluster，也就是聚类中心2范数的变化值，并通过delta_cluster_vec矢量存储所有变化值。
    delta_cluster = norm(tmp_center_cluster(1,:)-center_cluster(1,:))+norm(tmp_center_cluster(2,:)-center_cluster(2,:))+norm(tmp_center_cluster(3,:)-center_cluster(3,:))+norm(tmp_center_cluster(4,:)-center_cluster(4,:));
    center_cluster = tmp_center_cluster;
    delta_cluster_vec = [delta_cluster_vec delta_cluster];
end
%% 假定用最大隶属度表示归属，将各样本分类
class1=[];class2=[];class3=[];class4=[];
for numberi = 1:number
    tmp_index = find(degree(numberi,:)==max(degree(numberi,:)));
    if tmp_index == 1
        class1 = [class1 numberi];
    end
    if tmp_index == 2
        class2 = [class2 numberi];
    end 
    if tmp_index == 3
        class3 = [class3 numberi];
    end
    if tmp_index == 4
        class4 = [class4 numberi];
    end
end
%% 画图
subplot(1,2,1)
plot(sample(class1,1),sample(class1,2),'g+',sample(class2,1),sample(class2,2),'r+',sample(class3,1),sample(class3,2),'b+',sample(class4,1),sample(class4,2),'m+');
hold on
plot(center_cluster(1,1),center_cluster(1,2),'g*',center_cluster(2,1),center_cluster(2,2),'r*',center_cluster(3,1),center_cluster(3,2),'b*',center_cluster(4,1),center_cluster(4,2),'m*');
title('各点分类情况');
subplot(1,2,2)
plot(1:iteration,delta_cluster_vec,'k-');
set(gca,'XTick',1:1:iteration);
title('聚类中心模值变化随训练次数变化');