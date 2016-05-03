%  mian motivation
clearvars -except Dictionary;
clc;
load Dic_keda.mat;
SmoothMatrix=imread('lena.jpg');
input=double(rgb2gray(SmoothMatrix));
% data=input(231:370,241:380);
% [md, nd]=size(data);
% estimate_rank=rank(data);
% [u ,s, v]=svd(data);
% r=20;
% diags=diag(s);  %以S为主对角的矩阵
% [m, n]=size(s);
% %rs=length(diags);
% mu=diags(r);
% %diags=diags.*(diags>=mu);
% sc=s(:);
% sc=sc.*(sc>=mu);
% ss=reshape(sc,[m n]);
% img_gray=u*ss*v';
img_gray = input(231:370,241:380);
[col,row] = size(img_gray);
%%%---------------------------参数
lie = 30;
step = 25;
missing = 0.30; 
Lr=0.50; %整行的比例
lambda = 0.01;
%%%--------------------------mask =0为缺失
Dm = ones(col,row);
mline=missing*Lr;
smask = randperm(col,round(col*mline)); %先去掉整行
Dm(smask,:)=0;
ls = find(Dm~=0);
L= round(col*row*missing - round(col*mline)*row);
l_r = randsample(ls,L);
Dm(l_r) = 0;
img_noise = img_gray.*Dm;
%%
%%%---------------------------Relasp
num = floor((col-lie)/step);
yc = [1:step:num*step+1,col-lie+1]';
backA=zeros(col,row);
weight=zeros(col,row);
for i = 1:num+2
   y = yc(i);
   disp(['y = ',num2str(y),'进度 = ',num2str(i),'/',num2str(num+2)]);
   D = img_noise(y:y+lie-1,:);
   Dmask = Dm(y:y+lie-1,:);
   omega=find(Dmask~=0);
%---------------程明明的IALM
tic
[A,iter,svp] = inexact_alm_mc(D, 1e-4,700); 
    A=A.U*(A.V)';
output=A;
toc
% ----------SVT
% Dlie=D(omega);
% tic
% tau = 5*sqrt(lie*row)*12; %当数据归一化后，这个值要小一点，比如除25之类的。。。
% [U,S,V,numiter] = SVT([lie row],omega,Dlie,tau,1.5,500,1e-4);
% A = U*S*V';
% output=A;
% toc
%-----------------
%    [I,J]=ind2sub([lie row],omega);%121--49
%    [A,E,B]=IALM_reweighted_MC(D ,I , J, Dictionary, lambda , 1e-2, 700,1); %文献中重加权最多3次
%    output = D-E;
   backA(y:y+lie-1,:) = backA(y:y+lie-1,:)+output;
   weight(y:y+lie-1,:) = weight(y:y+lie-1,:)+1;
end
img_rec = backA./weight;
normlize = img_rec<0;
img_rec(normlize)=0;
normlize = img_rec>255;
img_rec(normlize)=255;
psnr = calcpsnr(img_gray,img_rec);
ssim = calcssim(img_rec,img_gray);
figure,imshow([uint8(img_gray),uint8(img_rec),uint8(img_noise)]);
title(['ori   &  rec  &  noise  PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim)]);
disp(['PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim)]);


