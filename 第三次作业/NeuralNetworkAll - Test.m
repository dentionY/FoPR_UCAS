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
All =  [A_add;B_add;C_add];  %30��4�о���
All_add = All([1:7 11:17 21:27],:); %21������
All_test = All([8:10 18:20 28:30],:);
Target = [ones(10,1) zeros(10,2);zeros(10,1) ones(10,1) zeros(10,1);zeros(10,2) ones(10,1)];
Target_add = Target([1:7 11:17 21:27],:);
Target_test = Target([8:10 18:20 28:30],:);
%% ��Ա���ʽȫ������������
%���������Ϊ���㣬����㡢�����㡢����㣻������ļ�����������tanh�����������ļ�����������sigmoid����;
theta = 0.01;%.................................................Jw�����ޣ��ɸĶ���
Input_I = 4;
Output_J = 3;
H = 4;%........................................................��������Ԫ�ĸ������ɸĶ���
eta = 0.1;%.....................................................ѧϰ�������ɸĶ���
w_ih = rand(Input_I,H)*2-1;%ƫ�õ�Ȩ��  %ÿ��ֵ�����i���������Ԫ����h����������Ԫ��Ȩֵ
w_hj = rand(H,Output_J)*2-1;                          %ÿ��ֵ�����h����������Ԫ����j���������Ԫ��Ȩֵ
Jw = 100;
delta_Jw = 100;
conditional_back = 100;
Jw_vector = [];
delta_Jw_vector = [];
conditional_back_vector = [];
%iteration = 0;
while conditional_back>theta
    %iteration = iteration + 1;
    Hidden_layer_add = zeros(21,H);  %30���������洢ÿ����������Ԫ��bias+��Ȩֵ
    Hidden_layer_sti = zeros(21,H);  %30���������洢ÿ����������Ԫ�ļ�������ֵ
    Out_layer_add = zeros(21,Output_J);%30���������������Ԫ�ļ�Ȩֵ
    Out_layer_sti = zeros(21,Output_J);%30���������������Ԫ�ļ���ֵ
    for Alli = 1:1:21
         %ǰ��ĵ�һ��������������õ���net_h�ͼ���ֵ����������tanh����
              for Hi = 1:1:H
                 Hidden_layer_add(Alli,Hi) = All_add(Alli,:)*w_ih(:,Hi);
                 exp_tmp_posi = exp(Hidden_layer_add(Alli,Hi));
                 exp_tmp_negd = exp(-1*Hidden_layer_add(Alli,Hi));                 
                 Hidden_layer_sti(Alli,Hi) = (exp_tmp_posi-exp_tmp_negd)/(exp_tmp_posi+exp_tmp_negd);
              end
         %ǰ��ĵڶ��������������õ���net_j�ͼ���ֵ�������Sigmoid����
              for Oi = 1:1:Output_J
                  Out_layer_add(Alli,Oi) = Hidden_layer_sti(Alli,:)*w_hj(:,Oi);
                  Out_layer_sti(Alli,Oi) = 1/(1+exp(-1*Out_layer_add(Alli,Oi)));
              end
          %ǰ��ĵ�����������Jw������Jw�кܶ࣬��˸������ֵѡ��Jw
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
    Jw_vector =[Jw_vector Jw]; %�洢ÿ��ѵ��������Jw
    delta_Jw_vector = [delta_Jw_vector delta_Jw]; %�洢ÿ��ѵ��������delta_Jw
   
    %����ĵ�0������һЩ���ݺ�Ԥ����
    Out_layer_compare = -Out_layer_sti+Target_add;%21��������ÿһ������Ԫ�ش����001/010/100�Ĳ��
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
            for Alli = 1:1:21   %�ۼ����е���������
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
            for Alli = 1:1:21
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

% �����ķ��������Է�������ѵ������
right = 0;
for togeri = 1:7
    if (find(Out_layer_sti(togeri,:) == max(Out_layer_sti(togeri,:))) == 1)
        right = right + 1;
    end
    if(find(Out_layer_sti(togeri+7,:) == max(Out_layer_sti(togeri+7,:))) == 2)
        right = right + 1;
    end
    if(find(Out_layer_sti(togeri+14,:) == max(Out_layer_sti(togeri+14,:))) == 3)
         right = right + 1;
    end
end
correct_rate = right/21;%---------------------------------------------------------------ѵ������ѵ�����ȣ���ȷ�ʣ�
false_rate = 1 - correct_rate;%---------------------------------------------------------ѵ�����Ĵ��������
explode = [1,0];
name = {'true','false'};
% �Բ��Եķ��������Է����������Ծ���
%�˴���Ҫǰ�����֮����ȷ�����
    THidden_layer_add = zeros(9,H);  %9���������洢ÿ����������Ԫ��bias+��Ȩֵ
    THidden_layer_sti = zeros(9,H);  %9���������洢ÿ����������Ԫ�ļ�������ֵ
    TOut_layer_add = zeros(9,Output_J);%9���������������Ԫ�ļ�Ȩֵ
    TOut_layer_sti = zeros(9,Output_J);%9���������������Ԫ�ļ���ֵ
    for TAlli = 1:1:9
         %ǰ��ĵ�һ��������������õ���net_h�ͼ���ֵ����������tanh����
              for THi = 1:1:H
                 THidden_layer_add(TAlli,THi) = All_test(TAlli,:)*w_ih(:,THi);
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
tright = 0;
for togeri = 1:3
    if (find(TOut_layer_sti(togeri,:) == max(TOut_layer_sti(togeri,:))) == 1)
        tright = tright + 1;
    end
    if(find(TOut_layer_sti(togeri+3,:) == max(TOut_layer_sti(togeri+3,:))) == 2)
        tright = tright + 1;
    end
    if(find(TOut_layer_sti(togeri+6,:) == max(TOut_layer_sti(togeri+6,:))) == 3)
         tright = tright + 1;
    end
end
Tcorrect_rate = tright/9;%------------------------------------------------------------���Լ��Ĳ��Ծ���
Tfalse_rate = 1 - Tcorrect_rate;%-----------------------------------------------------���Լ��Ĵ��������


figure(1)
subplot(2,2,1);
plot(Jw_vector([1:length(Jw_vector)]));title('�����������µ���ʧ������ѵ�������仯');ylabel('��ʧ����');xlabel('ѵ������');
subplot(2,2,2);
plot(conditional_back_vector([2:length(conditional_back_vector)]));title('�����������µ�ѭ�������仯��ѵ�������仯');ylabel('ѭ�������仯');xlabel('ѵ������');
subplot(2,2,3);
pie([correct_rate,false_rate],explode,name);title('ѵ��������ȷ����ı��ʺʹ������ı���');
subplot(2,2,4);
pie([Tcorrect_rate,Tfalse_rate],explode,name);title('����������ȷ����ı��ʺʹ������ı���');