
eta = 0.5;

X1=[0.1 6.8 -3.5 2.0 4.1 3.1 -0.8 0.9 5.0 3.9];
Y1=[1.1 7.1 -4.1 2.7 2.8 5.0 -1.3 1.2 6.4 4.0];
X2=[7.1 -1.4 4.5 6.3 4.2 1.4 2.4 2.5 8.4 4.1];
Y2=[4.2 -4.3 0.0 1.6 1.9 -3.2 -4.0 -6.1 3.7 -2.2];
X3=[-3.0 0.5 2.9 -0.1 -4.0 -1.3 -3.4 -4.1 -5.1 1.9];
Y3=[-2.9 8.7 2.1 5.2 2.2 3.7 6.2 3.4 1.6 5.1];
X4=[-2.0 -8.9 -4.2 -8.5 -6.7 -0.5 -5.3 -8.7 -7.1 -8.0];
Y4=[-8.4 0.2 -7.7 -3.2 -4.0 -9.2 -6.7 -6.4 -9.7 -6.3];
a = zeros(1,3);
%% 样本规范化，并统一归类到w1类，整合w2类样本
%{
Z1=ones(1,length(X1));
X2_normal = -X2;
Y2_normal = -Y2;
Z2_normal = -1.*ones(1,length(X2));
X_normal = [X1;X2_normal];
Y_normal = [Y1;Y2_normal];
Z_normal = [Z1;Z2_normal];
%}
%% 样本规范化，并统一归类到w3类，整合w2类样本

Z3=ones(1,length(X1));
X2_normal = -X2;
Y2_normal = -Y2;
Z2_normal = -1.*ones(1,length(X2));
X_normal = [X3;X2_normal];
Y_normal = [Y3;Y2_normal];
Z_normal = [Z3;Z2_normal];
gx_value = zeros(2,length(X3));

%% 样本规范化，并统一归类到w3类，整合w4类样本
%{
Z3=ones(1,length(X1));
X4_normal = -X4;
Y4_normal = -Y4;
Z4_normal = -1.*ones(1,length(X4));
X_normal = [X3;X4_normal];
Y_normal = [Y3;Y4_normal];
Z_normal = [Z3;Z4_normal];
gx_value = zeros(2,length(X3));
%}
%%
gx_value_neg = 1;  %因为要保证能进循环，所以设置为1
gx_value_neg_matr = zeros(2,length(X2));   %存储和a相乘之后的数值
iteration = 0;
gx_neg_vector = 0;
while gx_value_neg > 0
    gx_value_neg = 0;
    iteration = iteration + 1;
    a_tmp = a;
    for i = [1,2]
       %for j = 1:length(X2)
       for j=1:length(X2)
           gx_value(i,j) = a_tmp*[X_normal(i,j) Y_normal(i,j) Z_normal(i,j)].';
           if gx_value(i,j) <= 0  %寻找错分样本
              gx_value_neg = gx_value_neg + 1;
              gx_value_neg_matr(i,j) = 1;  %这一位是标记对应gx_value_neg矩阵数值的正负，0和负标记为1，正为0.
           else
               gx_value_neg_matr(i,j) = 0;
           end
           a = gx_value_neg_matr(i,j)*[X_normal(i,j) Y_normal(i,j) Z_normal(i,j)]*eta + a;
       end
    end
   gx_neg_vector = [gx_neg_vector gx_value_neg];
   % break
end
%plot([0:iteration],gx_neg_vector);
%%
figure(1)
bar(gx_neg_vector(2:length(gx_neg_vector)),0.5);
xlabel('迭代次数iteration');
ylabel('被分错的样本个数');
%title('w1和w2错分样本个数随迭代次数');
title('w3和w2错分样本个数随迭代次数');
%title('w3和w4错分样本个数随迭代次数');
%% 
%{
B=bar(barb1.');
grid on;
ch = get(B,'children');
xlabel('迭代次数iteration');
ylabel('错分样本个数');
title('两类训练错分样本个数随迭代次数变化');
%}
%legend('w1和w2分类','w3和w2分类');
figure(2)
%plot(X1,Y1,'+',X2,Y2,'O');
plot(X2,Y2,'+',X3,Y3,'O');
%plot3(X1,Y1,Z1,'*',X2,Y2,Z2,'o');
hold on
grid on
X=-10:.1:10;
%Y=(30.4*X-34)/34.1;
Y=(-a(1)*X-a(3))/a(2);
plot(X,Y);
title('二维平面下w3和w2两类点的分类直线');
%title('二维平面下w1和w2两类点的分类直线');
xlabel('x');ylabel('y');
%legend('w1','w2','w1和w2分界线');
legend('w3','w2','w3和w2分界线');
%legend('w3','w4','w3和w4分界线');
%[X,Y]=meshgrid(X);
figure(3)
%plot3(X3,Y3,Z3,'O',X4_normal,Y4_normal,Z4_normal,'*');
%plot3(X1,Y1,Z1,'O',X2_normal,Y2_normal,Z2_normal,'*');
plot3(X3,Y3,Z3,'O',X2_normal,Y2_normal,Z2_normal,'*');
hold on
xx=-10:.1:10;
yy=-10:.1:10;
[xx,yy]=meshgrid(xx,yy);
zz=-1/a(3)*(a(1)*xx+a(2)*yy);
mesh(xx,yy,zz);