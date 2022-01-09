
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
%% �����淶������ͳһ���ൽw1�࣬����w2������
%{
Z1=ones(1,length(X1));
X2_normal = -X2;
Y2_normal = -Y2;
Z2_normal = -1.*ones(1,length(X2));
X_normal = [X1;X2_normal];
Y_normal = [Y1;Y2_normal];
Z_normal = [Z1;Z2_normal];
%}
%% �����淶������ͳһ���ൽw3�࣬����w2������

Z3=ones(1,length(X1));
X2_normal = -X2;
Y2_normal = -Y2;
Z2_normal = -1.*ones(1,length(X2));
X_normal = [X3;X2_normal];
Y_normal = [Y3;Y2_normal];
Z_normal = [Z3;Z2_normal];
gx_value = zeros(2,length(X3));

%% �����淶������ͳһ���ൽw3�࣬����w4������
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
gx_value_neg = 1;  %��ΪҪ��֤�ܽ�ѭ������������Ϊ1
gx_value_neg_matr = zeros(2,length(X2));   %�洢��a���֮�����ֵ
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
           if gx_value(i,j) <= 0  %Ѱ�Ҵ������
              gx_value_neg = gx_value_neg + 1;
              gx_value_neg_matr(i,j) = 1;  %��һλ�Ǳ�Ƕ�Ӧgx_value_neg������ֵ��������0�͸����Ϊ1����Ϊ0.
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
xlabel('��������iteration');
ylabel('���ִ����������');
%title('w1��w2��������������������');
title('w3��w2��������������������');
%title('w3��w4��������������������');
%% 
%{
B=bar(barb1.');
grid on;
ch = get(B,'children');
xlabel('��������iteration');
ylabel('�����������');
title('����ѵ�����������������������仯');
%}
%legend('w1��w2����','w3��w2����');
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
title('��άƽ����w3��w2�����ķ���ֱ��');
%title('��άƽ����w1��w2�����ķ���ֱ��');
xlabel('x');ylabel('y');
%legend('w1','w2','w1��w2�ֽ���');
legend('w3','w2','w3��w2�ֽ���');
%legend('w3','w4','w3��w4�ֽ���');
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