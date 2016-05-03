%%% mian synthethic
close all;
clear all;
clc;
error = zeros(4,5);
for big=1:5 %7
m = 500;
% n = 200;
n = 2*m;
Dictionary =2*rand(m,n)-1; % dictionary  ��rand(m,n)��������-1��1��һ��
rank_patch = 0.05*m;  % rank
heng = m;
%============method B ��һ�ַ�������synthetic����,�����ձ�
B = zeros(n,rank_patch);%coe����-1��1��
for i=1:rank_patch
r = 7;%randi(10); % ϡ���
omega = randsample(n,r);
B(omega,i) = 2.*rand(r,1)-1; 
end
X = Dictionary*B;
R = rand(rank_patch,heng); %��randnҲ����̫�ֲ�
for i=1:heng
r = rank_patch-3;%randi(10); % 1-ϡ���
% r = 5+randi(10);
omega = randsample(rank_patch,r);
R(omega,i) = 0;
end
Low = X*R ; %Դ����
%Coe = B*R; ���յ�ϵ��������������ϡ���
%=========================
[lie,heng] = size(Low);
Dm = ones(lie,heng);
missing = 0.75; 
r=0.60; %���еı���
mline=missing*r;
smask = randperm(lie,round(lie*mline)); %��ȥ������
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
%=====================��������IALM
tic
[A1,iter,svp] = inexact_alm_mc(D, 1e-4,300); 
    A1=A1.U*(A1.V)';
output_IALM = A1;
toc
%=================SVT ��FPC
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
