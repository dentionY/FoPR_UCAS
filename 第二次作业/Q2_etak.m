X1=[0.1 6.8 -3.5 2.0 4.1 3.1 -0.8 0.9 5.0 3.9];
Y1=[1.1 7.1 -4.1 2.7 2.8 5.0 -1.3 1.2 6.4 4.0];
X2=[7.1 -1.4 4.5 6.3 4.2 1.4 2.4 2.5 8.4 4.1];
Y2=[4.2 -4.3 0.0 1.6 1.9 -3.2 -4.0 -6.1 3.7 -2.2];
X3=[-3.0 0.5 2.9 -0.1 -4.0 -1.3 -3.4 -4.1 -5.1 1.9];
Y3=[-2.9 8.7 2.1 5.2 2.2 3.7 6.2 3.4 1.6 5.1];

%% 样本规范化，并统一归类到w1类，整合w2类样本
Z1=ones(1,length(X1));
X2_normal = -X2;
Y2_normal = -Y2;
Z2_normal = -1.*ones(1,length(X2));
X_normal1 = [X1;X2_normal];
Y_normal1 = [Y1;Y2_normal];
Z_normal1 = [Z1;Z2_normal];

%% 样本规范化，并统一归类到w3类，整合w2类样本
Z3=ones(1,length(X1));
X2_normal = -X2;
Y2_normal = -Y2;
Z2_normal = -1.*ones(1,length(X2));
X_normal2 = [X3;X2_normal];
Y_normal2 = [Y3;Y2_normal];
Z_normal2 = [Z3;Z2_normal];

%%
etak = [0.1 0.5 1 5 10];
gx_value1 = zeros(2,length(X1));  %gx_value矩阵两行分别存储值是gx的判别值
gx_value2 = zeros(2,length(X3));

gx_value_neg_matr1 = zeros(2,length(X2));  %存储gx判别值是正还是非正，前者为0，后者为1
gx_value_neg_matr2 = zeros(2,length(X2));
iteration1 = zeros(1,length(etak));
iteration2 = zeros(1,length(etak));
gx_neg_vector1 = 0;
gx_neg_vector2 = 0;
for k = 1:length(etak)
  gx_value_neg1 = 1;  %因为要保证能进循环，所以设置为1
  a1 = zeros(1,3);
  while gx_value_neg1 > 0
    gx_value_neg1 = 0;
    a_tmp1 = a1;
    iteration1(1,k) = iteration1(1,k) + 1;
      for i = [1,2]
        for j = 1:length(X2)
               gx_value1(i,j) = a_tmp1*[X_normal1(i,j) Y_normal1(i,j) Z_normal1(i,j)].';
               if gx_value1(i,j) <= 0  %寻找错分样本
                  gx_value_neg1 = gx_value_neg1 + 1;
                  gx_value_neg_matr1(i,j) = 1;  %这一位是标记对应gx_value_neg矩阵数值的正负，0和负标记为1，正为0.
               else
                  gx_value_neg_matr1(i,j) = 0;
               end
           a1 = gx_value_neg_matr1(i,j)*[X_normal1(i,j) Y_normal1(i,j) Z_normal1(i,j)]*etak(k) + a1;
        end
     end
   gx_neg_vector1 = [gx_neg_vector1 gx_value_neg1];
  end
   % break
end
for k = 1:length(etak)
  gx_value_neg2 = 1;  %因为要保证能进循环，所以设置为1
  a2 = zeros(1,3);
  while gx_value_neg2 > 0
    gx_value_neg2 = 0;
    iteration2(1,k) = iteration2(1,k) + 1;
    a_tmp2 = a2;
    for i = [1,2]
       for j = 1:length(X2)
           gx_value2(i,j) = a_tmp2*[X_normal2(i,j) Y_normal2(i,j) Z_normal2(i,j)].';
           if gx_value2(i,j) <= 0  %寻找错分样本
              gx_value_neg2 = gx_value_neg2 + 1;
              gx_value_neg_matr2(i,j) = 1;  %这一位是标记对应gx_value_neg矩阵数值的正负，0和负标记为1，正为0.
           else
               gx_value_neg_matr2(i,j) = 0;
           end
           a2 = gx_value_neg_matr2(i,j)*[X_normal2(i,j) Y_normal2(i,j) Z_normal2(i,j)]*etak(k) + a2;
       end
    end
   gx_neg_vector2 = [gx_neg_vector2 gx_value_neg2];
  end
   % break
end
%plot([0:iteration],gx_neg_vector);
iteration = [iteration1;iteration2];
%% 
B=bar(iteration.');
grid on;
ch = get(B,'children');
set(gca,'XTickLabel',{'0.1','0.5','1','5','10'});
%set(gca,'XTickLabel',{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'});
%set(ch,'FaceVertexCData',[1 0 1;0 0 0;])
xlabel('hn');
ylabel('训练次数');
title('两类训练次数随窗宽变化');
legend('w1和w2分类','w3和w2分类');