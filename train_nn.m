function[Wold1, Wold2] =train_nn()

load('Project2_dataset.mat');

% d0=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\zero.xlsx');
t0=zeros(size(d0,1),10);
t0(:,1)=1;

%d1=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\ones.xlsx');
t1=zeros(size(d1,1),10);
t1(:,2)=1;

%d2=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\two.xlsx');
t2=zeros(size(d2,1),10);
t2(:,3)=1;

%d3=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\three.xlsx');
t3=zeros(size(d3,1),10);
t3(:,4)=1;

%d4=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\four.xlsx');
t4=zeros(size(d4,1),10);
t4(:,5)=1;

%d5=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\five.xlsx');
t5=zeros(size(d5,1),10);
t5(:,6)=1;

%d6=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\six.xlsx');
t6=zeros(size(d6,1),10);
t6(:,7)=1;

%d7=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\seven.xlsx');
t7=zeros(size(d7,1),10);
t7(:,8)=1;

%d8=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\eight.xlsx');
t8=zeros(size(d8,1),10);
t8(:,9)=1;

%d9=xlsread('C:\Users\asles_000\Desktop\SEM_1\ML\PROJECT\PROJECT 2\mydata\nine.xlsx');
t9=zeros(size(d9,1),10);
t9(:,10)=1;



t=[t0; t1; t2; t3; t4; t5; t6; t7; t8; t9];
fi=[d0; d1; d2; d3; d4; d5; d6; d7; d8; d9];  
N=size(fi,1);
fi=[ones(N,1) fi];    

M=20;
D=513;
n1=0.0001;
n2=0.0001;

K=10;

Wji=rand(M, D)-0.5;
Wkj=rand(K, M+1)-0.5;

E=inf;

for k=1:2000

 hold on;
 plot(k,E);
    
Wold2=Wkj;
Wold1=Wji;

aj= fi*Wji';
zj_old=tanh(aj);

zj=[ones(N,1) zj_old]; %adding bias

ak=zj*Wkj';
ak_exp=exp(ak);
ak_sum=sum(ak_exp,2);

P=bsxfun(@rdivide, ak_exp, ak_sum);

%cross entropy error
C_Error=0;
for i=1:N
   
        C_Error=C_Error+(-1)*dot(t(i,:),log(P(i,:)));
end
E=C_Error
k

% error backpropagation
delk=P-t;
grad_error2=delk' * zj;

% Compute SUM(del_k * w_kj) over k
    delj_with_bias = delk * Wkj;
    delj = delj_with_bias(:,2:end);
    
  grad_error1 = (delj .* (ones(N,M) - zj_old.^2))' * fi;  
    
  
 % weight update stage
 Wji=Wji-n1*grad_error1;
 Wkj=Wkj-n2*grad_error2;

 if(E<300)
     Wji=Wold1;
     Wkj=Wold2;
     break;
 end
 
end
 