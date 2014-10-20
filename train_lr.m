function[W]=train_lr()

load('Project2_dataset.mat');

t0=zeros(size(d0,1),10);
t0(:,1)=1;

t1=zeros(size(d1,1),10);
t1(:,2)=1;

t2=zeros(size(d2,1),10);
t2(:,3)=1;

t3=zeros(size(d3,1),10);
t3(:,4)=1;

t4=zeros(size(d4,1),10);
t4(:,5)=1;

t5=zeros(size(d5,1),10);
t5(:,6)=1;

t6=zeros(size(d6,1),10);
t6(:,7)=1;

t7=zeros(size(d7,1),10);
t7(:,8)=1;

t8=zeros(size(d8,1),10);
t8(:,9)=1;

t9=zeros(size(d9,1),10);
t9(:,10)=1;


t=[t0; t1; t2; t3; t4; t5; t6; t7; t8; t9];
fi=[d0; d1; d2; d3; d4; d5; d6; d7; d8; d9];  
N=size(fi,1);
fi=[ones(N,1) fi];    

W=rand(513,10);
E=inf;
for k=1:2000

hold on;
plot(k,E);
    
a=fi*W;
p=exp(a);
s=sum(p,2);

P=bsxfun(@rdivide, p, s);

 temp=P-t;
 gradient_error=fi'*temp;
 

Wnew=W-(0.0005*gradient_error);

C_Error=0;
for i=1:N
   
        C_Error=C_Error+(-1)*dot(t(i,:),log(P(i,:)));
end
E=C_Error
k
if(E<300) 
    break;
end
W=Wnew;
end
