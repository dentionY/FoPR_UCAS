%t1 = clock;
A=[1.58,2.32 -5.8;0.67,1.58,-4.78;1.04,1.01,-3.63;
   -1.49,2.18,-3.39;-0.41,1.21,-4.73;1.39,3.16,2.87;
   1.20,1.40,-1.89;-0.92,1.44,-3.22;0.45,1.33,-4.38;
   -0.76,0.84,-1.96];
B=[0.21, 0.03, -2.21;0.37, 0.28, -1.8;0.18, 1.22, 0.16;
   -0.24, 0.93, -1.01;-1.18, 0.39, -0.39;0.74, 0.96, -1.16;
   -0.38, 1.94, -0.48;0.02, 0.72, -0.17;0.44, 1.31, -0.14;
   0.46, 1.49, 0.68];
C=[-1.54, 1.17, 0.64;5.41, 3.45, -1.33;1.55, 0.99, 2.69;
   1.86, 3.19, 1.51;1.68, 1.79, -0.87;3.51, -0.22, -1.39;
   1.40, -0.44, -0.92;0.44, 0.83, 1.97;0.25, 0.68, -0.99;
   0.66, -0.45, 0.08];
%增加偏置
A_add = [A ones(10,1)];  %10行4列矩阵
B_add = [B ones(10,1)];
C_add = [C ones(10,1)];
All_add = [A_add;B_add;C_add];  %30行4列矩阵
Target_add = [ones(10,1) zeros(10,2);zeros(10,1) ones(10,1) zeros(10,1);zeros(10,2) ones(10,1)];
%% 针对遍历式全样本更新网络
%其中网络分为三层，输入层、隐含层、输出层；隐含层的激励函数采用tanh函数；输出层的激励函数采用sigmoid函数;
theta = 0.01;%Jw的上限
Input_I = 4;
Output_J = 3;
H = 10;%隐含层神经元的个数
eta = 0.1;
w_ih = rand(Input_I,H)*2-1;%偏置的权重  %每个值代表第i个输入层神经元到第h个隐含层神经元的权值
w_hj = rand(H,Output_J)*2-1;                          %每个值代表第h个隐含层神经元到第j个输出层神经元的权值
Jw = 100;
delta_Jw = 100;
conditional_back = 100;
Jw_vector = [];
delta_Jw_vector = [];
conditional_back_vector = [];
%iteration = 0;
while conditional_back>theta
    %iteration = iteration + 1;
    Hidden_layer_add = zeros(30,H);  %30个样本，存储每个隐含层神经元的bias+加权值
    Hidden_layer_sti = zeros(30,H);  %30个样本，存储每个隐含层神经元的激励函数值
    Out_layer_add = zeros(30,Output_J);%30个样本，输出层神经元的加权值
    Out_layer_sti = zeros(30,Output_J);%30个样本，输出层神经元的激励值
    for Alli = 1:1:30
         %前向的第一步，计算隐含层得到的net_h和激励值，隐含层用tanh函数
              for Hi = 1:1:H
                 Hidden_layer_add(Alli,Hi) = All_add(Alli,:)*w_ih(:,Hi);
                 exp_tmp_posi = exp(Hidden_layer_add(Alli,Hi));
                 exp_tmp_negd = exp(-1*Hidden_layer_add(Alli,Hi));                 
                 Hidden_layer_sti(Alli,Hi) = (exp_tmp_posi-exp_tmp_negd)/(exp_tmp_posi+exp_tmp_negd);
              end
         %前向的第二步，计算输出层得到的net_j和激励值，输出用Sigmoid函数
              for Oi = 1:1:Output_J
                  Out_layer_add(Alli,Oi) = Hidden_layer_sti(Alli,:)*w_hj(:,Oi);
                  Out_layer_sti(Alli,Oi) = 1/(1+exp(-1*Out_layer_add(Alli,Oi)));
              end
          %前向的第三步，更新Jw，由于Jw有很多，因此根据最大值选择Jw
              minus_vec_tmp = (Out_layer_sti(Alli,:)-Target_add(Alli,:)).*(Out_layer_sti(Alli,:)-Target_add(Alli,:));
              Jw_temp = sum(minus_vec_tmp)/2;
              if Alli == 1
                  Jw = Jw_temp;
                  delta_Jw = 100;
              else
                  if Jw < Jw_temp
                      delta_Jw = abs(Jw-Jw_temp);
                      Jw = Jw_temp;
                  end
              end
    end
    Jw_vector =[Jw_vector Jw]; %存储每次训练次数的Jw
    delta_Jw_vector = [delta_Jw_vector delta_Jw]; %存储每次训练次数的delta_Jw
   
    %反向的第0步，做一些备份和预计算
    Out_layer_compare = -Out_layer_sti+Target_add;%30个样本，每一行三个元素代表和001/010/100的差距
    w_hj_backoff = w_hj;
    w_ih_backoff = w_ih;
    conditional_back1 = 0;
    conditional_back2 = 0;
    %反向的第一步，更新w_hj
    for Oi = 1:1:Output_J
        conditional_back1_retmp = 0;
        for Hi = 1:1:H
            delta_whj = 0;
            conditional_back1_tmp = 0;
            for Alli = 1:1:30   %累加所有的样本贡献
                delta_whj = delta_whj + eta*Out_layer_compare(Alli,Oi)*Out_layer_sti(Alli,Oi)*(1-Out_layer_sti(Alli,Oi))*Hidden_layer_sti(Alli,Hi);
                conditional_back1_tmp = conditional_back1_tmp + power(abs(Out_layer_compare(Alli,Oi)*Out_layer_sti(Alli,Oi)*(1-Out_layer_sti(Alli,Oi))*Hidden_layer_sti(Alli,Hi)),2);
            end
            w_hj(Hi,Oi) = w_hj(Hi,Oi) + delta_whj;
            conditional_back1_retmp = conditional_back1_retmp + conditional_back1_tmp;
        end
        conditional_back1 = conditional_back1 + conditional_back1_retmp;
    end
    %反向的第二步，更新w_ih
    for Hi = 1:1:H
        conditional_back2_retmp = 0;
        for Ii = 1:1:Input_I
            delta_wih = 0;
            conditional_back2_tmp = 0;
            for Alli = 1:1:30
                delta_wih_Jpart = 0;
                for Oi = 1:1:Output_J
                    delta_wih_Jpart = delta_wih_Jpart + Out_layer_compare(Alli,Oi)*Out_layer_sti(Alli,Oi)*(1-Out_layer_sti(Alli,Oi))*w_hj_backoff(Hi,Oi);
                end
                delta_wih = delta_wih + delta_wih_Jpart*eta*(1-Hidden_layer_sti(Alli,Hi)*Hidden_layer_sti(Alli,Hi))*All_add(Alli,Ii);
                conditional_back2_tmp = conditional_back2_tmp + power(abs(delta_wih_Jpart*(1-Hidden_layer_sti(Alli,Hi)*Hidden_layer_sti(Alli,Hi))*All_add(Alli,Ii)),2);
            end
            w_ih(Ii,Hi) = w_ih(Ii,Hi) + delta_wih;
            conditional_back2_retmp = conditional_back2_retmp + conditional_back2_tmp;
        end
        conditional_back2 = conditional_back2 + conditional_back1_retmp;
    end
    conditional_back = sqrt(conditional_back1 + conditional_back2);
    conditional_back_vector = [conditional_back_vector conditional_back];
end

% 对最后的分类结果加以分析
right = 0;
for togeri = 1:10
    if (find(Out_layer_sti(togeri,:) == max(Out_layer_sti(togeri,:))) == 1)
        right = right + 1;
    end
    if(find(Out_layer_sti(togeri+10,:) == max(Out_layer_sti(togeri+10,:))) == 2)
        right = right + 1;
    end
    if(find(Out_layer_sti(togeri+20,:) == max(Out_layer_sti(togeri+20,:))) == 3)
         right = right + 1;
    end
end
correct_rate = right/30;
false_rate = 1 - correct_rate;
explode = [1,0];
name = {'true','false'};

figure(1)
subplot(2,2,1);
plot(Jw_vector([1:length(Jw_vector)]));title('批量样本更新的损失函数随训练次数变化');ylabel('损失函数');xlabel('训练次数');
subplot(2,2,2);
plot(conditional_back_vector([2:length(conditional_back_vector)]));title('批量样本更新的循环条件变化随训练次数变化');ylabel('循环条件变化');xlabel('训练次数');
subplot(2,2,[3,4]);
pie([correct_rate,false_rate],explode,name);title('正确分类的比率和错误分类的比率');