%%% mian color image
close all;
clear all;
clc;
tol = 1e-4;
tu = zeros(50,50);
%%
start_missing=0.15 %起始点，可以先自己估测，逐渐调试到下三角区域就行。RE从0.2开始
for rank_patch = 50:-1:1  % rank
    cnt_r = rank_patch;
 for missing = start_missing:0.01:0.5
      sr=0; %整行的比例
      cnt_m = uint8(51-missing*100);
      disp([cnt_r,cnt_m]); % 在这设置一个起点记录标志
yesno = zeros(5,1);
for big=1:5
m = 100;
n = 200;
Dictionary =2*rand(m,n)-1; % dictionary  用rand(m,n)不控制在-1～1好一点
% rank_patch = 0.05*m;  % rank
heng = m;
%============method B 另一种方法产生synthetic数据,更加普遍
B = zeros(n,rank_patch);%coe，【-1，1】
for i=1:rank_patch
r = 5;%randi(10); % 稀疏度
omega = randsample(n,r);
B(omega,i) = 2.*rand(r,1)-1; 
end
X = Dictionary*B;
R = rand(rank_patch,heng); %用randn也行正太分布
for i=1:heng
r = rank_patch-4;%randi(10); % 1-稀疏度
% r = 5+randi(10);
omega = randsample(rank_patch,r);
R(omega,i) = 0;
end
Low = X*R ; %源数据
%Coe = B*R; 最终的系数，可以用于算稀疏度
%=========================
[lie,heng] = size(Low);
Dm = creatmask(lie,heng,missing,sr);
D=Low.*Dm;
omega=find(Dm~=0);
[I,J]=ind2sub([lie heng],omega);%121--49
Dlie=D(omega);
%=======================our method
[~,E,~]=IALM_reweighted_MC(D ,I , J, Dictionary, 1 , 1e-7, 500,3); 
output = D-E;
%=====================程明明的IALM
% tic
% [A1,iter,svp] = inexact_alm_mc(D, 1e-7,500); 
%  output=A1.U*(A1.V)';
% toc
%=================SVT 和FPC
% tic
% tau = 5*sqrt(lie*heng)/10;
% [U,S,V,numiter] = SVT([lie heng],omega,Dlie,tau,1.5,500,1e-7);
% output = U*S*V';
% toc
% ----------------------------------
% output = zeros(lie,heng,3);
% output(:,:,1)= output_SVT;
% output(:,:,2)= output_IALM;
% output= output_RE2;

    yesno(big,1) = norm(Low-output,'fro')/norm(Low,'fro');
   if yesno(big,1) < tol
        yesno(big,1) = 1;
   else
       yesno(big,1) = 0;
   end

end
results = sum(yesno,1)/5; %
if results == 1
 start_missing = missing;  %可以再加0.01
end

tu(cnt_m,cnt_r) = results;  %记录结果，<0.1 则跳出本rank的循环
 if results < 0.1
 break;
 end
 
 end 
end
imagesc(tu);
colormap(gray);
save tu_RE_5-4-middle.mat
%得到的tu是没有处理成下三角为1的矩阵，如果显示还需要处理。
% flipud 对称矩阵