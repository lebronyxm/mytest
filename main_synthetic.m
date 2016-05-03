%%% mian synthethic
close all;
clear all;
clc;
error = zeros(4,5);
for big=1:5 %7
m = 500;
% n = 200;
n = 2*m;
Dictionary =2*rand(m,n)-1; % dictionary  用rand(m,n)不控制在-1～1好一点
rank_patch = 0.05*m;  % rank
heng = m;
%============method B 另一种方法产生synthetic数据,更加普遍
B = zeros(n,rank_patch);%coe，【-1，1】
for i=1:rank_patch
r = 7;%randi(10); % 稀疏度
omega = randsample(n,r);
B(omega,i) = 2.*rand(r,1)-1; 
end
X = Dictionary*B;
R = rand(rank_patch,heng); %用randn也行正太分布
for i=1:heng
r = rank_patch-3;%randi(10); % 1-稀疏度
% r = 5+randi(10);
omega = randsample(rank_patch,r);
R(omega,i) = 0;
end
Low = X*R ; %源数据
%Coe = B*R; 最终的系数，可以用于算稀疏度
%=========================
[lie,heng] = size(Low);
Dm = ones(lie,heng);
missing = 0.75; 
r=0.60; %整行的比例
mline=missing*r;
smask = randperm(lie,round(lie*mline)); %先去掉整行
Dm(smask,:)=0;
ls = find(Dm~=0);
L= round(lie*heng*missing - round(lie*mline)*heng);
lr = randsample(ls,L);
Dm(lr) = 0;
%%
D=Low.*Dm;
omega=find(Dm~=0);
[I,J]=ind2sub([lie heng],omega);%121--49
Dlie=D(omega);
%=======================our method
[~,E,~]=IALM_reweighted_MC(D ,I , J, Dictionary, 1 , 1e-4, 300,3); 
output_RE = D-E(:,:,1);
output_RE2 = D-E(:,:,3);
%=====================程明明的IALM
tic
[A1,iter,svp] = inexact_alm_mc(D, 1e-4,300); 
    A1=A1.U*(A1.V)';
output_IALM = A1;
toc
%=================SVT 和FPC
tic
tau = 5*sqrt(lie*heng)/10;
[U,S,V,numiter] = SVT([lie heng],omega,Dlie,tau,1.5,300,1e-4);
A2 = U*S*V';
output_SVT = A2;
toc
% ----------------------------------
output = zeros(lie,heng,4);
output(:,:,1)= output_SVT;
output(:,:,2)= output_IALM;
output(:,:,3)= output_RE;
output(:,:,4)= output_RE2;
% fd_txt        =   fopen( 'C:\Users\Administrator\Desktop\synthetic_re.txt', 'wt');

for it=1:4
    error(it,big) = norm(Low-output(:,:,it),'fro')/norm(Low,'fro');
%     fprintf(fd_txt, 'hh%s :    RE = %2.4f\n', num2str(it), error(it));
end
% fclose(fd_txt);
end
err_mean = mean(error,2);
imshow(it);
