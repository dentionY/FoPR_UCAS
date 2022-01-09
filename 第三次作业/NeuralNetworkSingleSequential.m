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
%����ƫ��
A_add = [A ones(10,1)];  %10��4�о���
B_add = [B ones(10,1)];
C_add = [C ones(10,1)];
All_add = [A_add;B_add;C_add];  %30��4�о���
Target_add = [ones(10,1) zeros(10,2);zeros(10,1) ones(10,1) zeros(10,1);zeros(10,2) ones(10,1)];
%% ���˳��������������
%���������Ϊ���㣬����㡢�����㡢����㣻������ļ�����������tanh�����������ļ�����������sigmoid����;
theta = 0.01;%Jw������
Input_I = 4;
Output_J = 3;
H = 10;%��������Ԫ�ĸ���
w_ih = rand(Input_I,H)*2-1;  %ÿ��ֵ�����i���������Ԫ����h����������Ԫ��Ȩֵ
w_hj = rand(H,Output_J)*2-1;                          %ÿ��ֵ�����h����������Ԫ����j���������Ԫ��Ȩֵ
Jw = 100;
delta_Jw = 100;
conditional_back = 100;
Jw_vector = [];
delta_Jw_vector = [];
conditional_back_vector = [];
eta = 0.1;
iteration = 0;
while conditional_back>theta
    iteration = iteration + 1;
    Hidden_layer_add = zeros(1,H);  %1���������洢ÿ����������Ԫ��bias+��Ȩֵ
    Hidden_layer_sti = zeros(1,H);  %1���������洢ÿ����������Ԫ�ļ�������ֵ
    Out_layer_add = zeros(1,Output_J);%1���������������Ԫ�ļ�Ȩֵ
    Out_layer_sti = zeros(1,Output_J);%1���������������Ԫ�ļ���ֵ
    %ͨ��˳��ѭ���õ�ÿ��ѵ���õ��������
    sequentialindex_tmp = mod(iteration,30);
    if sequentialindex_tmp == 0
       sequentialindex = 30;
    else
       sequentialindex = sequentialindex_tmp;
    end
    for Alli = 1
         %ǰ��ĵ�һ��������������õ���net_h�ͼ���ֵ����������tanh����
              for Hi = 1:1:H
                 Hidden_layer_add(Alli,Hi) = All_add(sequentialindex,:)*w_ih(:,Hi);
                 exp_tmp_posi = exp(Hidden_layer_add(Alli,Hi));
                 exp_tmp_negd = exp(-1*Hidden_layer_add(Alli,Hi));                 
                 Hidden_layer_sti(Alli,Hi) = (exp_tmp_posi-exp_tmp_negd)/(exp_tmp_posi+exp_tmp_negd);
              end
         %ǰ��ĵڶ��������������õ���net_j�ͼ���ֵ�������Sigmoid����
              for Oi = 1:1:Output_J
                  Out_layer_add(Alli,Oi) = Hidden_layer_sti(Alli,:)*w_hj(:,Oi);
                  Out_layer_sti(Alli,Oi) = 1/(1+exp(-1*Out_layer_add(Alli,Oi)));
              end
          %ǰ��ĵ�����������Jw
              minus_vec_tmp = (Out_layer_sti(Alli,:)-Target_add(sequentialindex,:)).*(Out_layer_sti(Alli,:)-Target_add(sequentialindex,:));
              Jw_tmp = sum(minus_vec_tmp)/2;
             if iteration == 1
                 delta_Jw = Jw_tmp;
                 Jw = Jw_tmp;
             else
                delta_Jw = abs(Jw_tmp - Jw);
                Jw = Jw_tmp;
             end
    end
    Jw_vector =[Jw_vector Jw]; %�洢ÿ��ѵ��������Jw
    delta_Jw_vector = [delta_Jw_vector delta_Jw];
   
    %����ĵ�0������һЩ���ݺ�Ԥ����
    Out_layer_compare = -Out_layer_sti+Target_add(sequentialindex,:);%1��������ÿһ������Ԫ�ش����001/010/100�Ĳ��
    w_hj_backoff = w_hj;
    w_ih_backoff = w_ih;
    conditional_back1 = 0;
    conditional_back2 = 0;
    %����ĵ�һ��������w_hj
    for Oi = 1:1:Output_J
        conditional_back1_retmp = 0;
        for Hi = 1:1:H
            delta_whj = 0;
            conditional_back1_tmp = 0;
            for Alli = 1   %�ۼ����е���������
                delta_whj = delta_whj + eta*Out_layer_compare(Alli,Oi)*Out_layer_sti(Alli,Oi)*(1-Out_layer_sti(Alli,Oi))*Hidden_layer_sti(Alli,Hi);
                conditional_back1_tmp = conditional_back1_tmp + power(abs(Out_layer_compare(Alli,Oi)*Out_layer_sti(Alli,Oi)*(1-Out_layer_sti(Alli,Oi))*Hidden_layer_sti(Alli,Hi)),2);
            end
            w_hj(Hi,Oi) = w_hj(Hi,Oi) + delta_whj;
            conditional_back1_retmp = conditional_back1_retmp + conditional_back1_tmp;
        end
        conditional_back1 = conditional_back1 + conditional_back1_retmp;
    end
    %����ĵڶ���������w_ih
    for Hi = 1:1:H
        conditional_back2_retmp = 0;
        for Ii = 1:1:Input_I
            delta_wih = 0;
            conditional_back2_tmp = 0;
            for Alli = 1
                delta_wih_Jpart = 0;
                for Oi = 1:1:Output_J
                    delta_wih_Jpart = delta_wih_Jpart + Out_layer_compare(Alli,Oi)*Out_layer_sti(Alli,Oi)*(1-Out_layer_sti(Alli,Oi))*w_hj_backoff(Hi,Oi);
                end
                delta_wih = delta_wih + delta_wih_Jpart*eta*(1-Hidden_layer_sti(Alli,Hi)*Hidden_layer_sti(Alli,Hi))*All_add(sequentialindex,Ii);
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
%%  ���ڼ�����������Ƿ������ȷ
    THidden_layer_add = zeros(30,H);  %30���������洢ÿ����������Ԫ��bias+��Ȩֵ
    THidden_layer_sti = zeros(30,H);  %30���������洢ÿ����������Ԫ�ļ�������ֵ
    TOut_layer_add = zeros(30,Output_J);%30���������������Ԫ�ļ�Ȩֵ
    TOut_layer_sti = zeros(30,Output_J);%30���������������Ԫ�ļ���ֵ
    for TAlli = 1:1:30
         %ǰ��ĵ�һ��������������õ���net_h�ͼ���ֵ����������tanh����
              for THi = 1:1:H
                 THidden_layer_add(TAlli,THi) = All_add(TAlli,:)*w_ih(:,THi);
                 Texp_tmp_posi = exp(THidden_layer_add(TAlli,THi));
                 Texp_tmp_negd = exp(-1*THidden_layer_add(TAlli,Hi));                 
                 THidden_layer_sti(TAlli,Hi) = (Texp_tmp_posi-Texp_tmp_negd)/(Texp_tmp_posi+Texp_tmp_negd);
              end
         %ǰ��ĵڶ��������������õ���net_j�ͼ���ֵ�������Sigmoid����
              for TOi = 1:1:Output_J
                  TOut_layer_add(TAlli,TOi) = THidden_layer_sti(TAlli,:)*w_hj(:,TOi);
                  TOut_layer_sti(TAlli,TOi) = 1/(1+exp(-1*TOut_layer_add(TAlli,TOi)));
              end
    end
right = 0;
for togeri = 1:10
    if (find(TOut_layer_sti(togeri,:) == max(TOut_layer_sti(togeri,:))) == 1)
        right = right + 1;
    end
    if(find(TOut_layer_sti(togeri+10,:) == max(TOut_layer_sti(togeri+10,:))) == 2)
        right = right + 1;
    end
    if(find(TOut_layer_sti(togeri+20,:) == max(TOut_layer_sti(togeri+20,:))) == 3)
         right = right + 1;
    end
end
correct_rate = right/30;
false_rate = 1 - correct_rate;
explode = [1,0];
name = {'true','false'};

figure(1)
subplot(2,2,1);
plot(Jw_vector([1:length(Jw_vector)]));title('����ʽ���������µ���ʧ������ѵ�������仯');ylabel('��ʧ����');xlabel('ѵ������');
subplot(2,2,2);
plot(conditional_back_vector([2:length(conditional_back_vector)]));title('����ʽ���������µ�ѭ�������仯��ѵ�������仯');ylabel('ѭ�������仯');xlabel('ѵ������');
subplot(2,2,[3,4]);
pie([correct_rate,false_rate],explode,name);title('��ȷ����ı��ʺʹ������ı���');