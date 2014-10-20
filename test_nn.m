function [error_rate, labels] = test_nn()

[Wold1, Wold2]=train_nn();

load('Project2_dataset.mat');

 ttest0=zeros(size(dtest0,1),10);
 ttest0(:,1)=1;
 
 ttest1=zeros(size(dtest1,1),10);
 ttest1(:,2)=1;
 
 ttest2=zeros(size(dtest2,1),10);
 ttest2(:,3)=1;

ttest3=zeros(size(dtest3,1),10);
ttest3(:,4)=1;

ttest4=zeros(size(dtest4,1),10);
ttest4(:,5)=1;

ttest5=zeros(size(dtest5,1),10);
ttest5(:,6)=1;

ttest6=zeros(size(dtest6,1),10);
ttest6(:,7)=1;

ttest7=zeros(size(dtest7,1),10);
ttest7(:,8)=1;

ttest8=zeros(size(dtest8,1),10);
ttest8(:,9)=1;

ttest9=zeros(size(dtest9,1),10);
ttest9(:,10)=1;

ttest=[ttest0; ttest1; ttest2; ttest3; ttest4; ttest5; ttest6; ttest7; ttest8; ttest9];

fi_test=[dtest0; dtest1; dtest2; dtest3; dtest4; dtest5; dtest6; dtest7; dtest8; dtest9];
 N=size(fi_test,1);

 fi_test=[ones(N,1) fi_test]; %adding bias row to phi 

 %training the training data
 
Wji=Wold1;
Wkj=Wold2;

%input to the hidden layer
aj= fi_test*Wji';
zj_old=tanh(aj);

zj=[ones(N,1) zj_old]; %adding bias


%Using Softmax function to calculate probablities
ak=zj*Wkj';
ak_exp=exp(ak);
ak_sum=sum(ak_exp,2);

P=bsxfun(@rdivide, ak_exp, ak_sum);

%Counting number of misclassifications
error_count=0;
 
[r, x]=max(P,[],2);
P(:,:)=0;
for i=1:N
    P(i,x(i))=1;
    if(isequal(P(i,:),ttest(i,:))~=true)
        error_count=error_count+1;
    end
end

%Storing predicted class labels
x=x-1;
dlmwrite('classes_nn.txt',x);
labels=x;

error_rate=(error_count/N)*100;
end