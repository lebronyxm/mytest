%inpainting main - single image
% clearvars -except ori_img;
clc;
% load Dic_36-500.mat;
% Dictionary=Dictionary/10;
% Pn=ceil(sqrt(70));   %K表示字典的大小,K取324,是平方
%      %初始化一个扁矩阵，bb为字典原子维数开平方，如你要产生256为的，则设置bb=16；
%  bb=5;
%  DCT=zeros(bb,Pn);
%  for k=0:1:Pn-1,     
%     V=cos([0:1:bb-1]'*k*pi/Pn);
%     if k>0
%         V=V-mean(V); 
%     end;
%     DCT(:,k+1)=V/norm(V);
%  end;
%  Dictionary=kron(DCT,DCT)*14;%G就是最终的DCT字典； *14
%Dictionary=Dictionary;
% load ('F:\programs\matlab\myprogram\Dm.mat');
img_rgb  = imread('G:\picset\test\lena.png'); %G:\picset\test\lena.png
% img_rgb = img_rgb(100:250,180:180+160);
% img_gray = double(rgb2gray(img_rgb));
img_gray = double(img_rgb);
[col,row] = size(img_gray);
%%%---------------------------参数
lie = 30;
step = 10;
missing = 0.20; 
Lr=0.30; %整行的比例
noise_gs = 20;
noise_sp = 0.3;
%%%--------------------------mask =0为缺失
Dm = creatmask(col,row,missing,Lr);
% img_gs = Add_noise_sparse (img_gray,noise_sp); 
img_gs = Add_noise(img_gray,noise_gs);
% img_noise = img_gray.*Dm;
img_noise = img_gs.*Dm;
% img_noise = double(imread('G:\picset\test\lena_2.pgm'));
%%%---------------------------iteration
%%
num = floor((col-lie)/step);
yc = [1:step:num*step+1,col-lie+1]';
backA=zeros(col,row);
weight=zeros(col,row);
%------10b倍DCT字典
%---下面是100×512的
% a=0.5; %1
% b=0.5; %0.04,0.03  last0.4
% e=0.003; % 0.1
%---------下面的是36×512的
a=0.4; %1
b=0.01; %0.04,0.03
e=0.3; % 0.1
%----------下面是训练字典
% a=1.2; %1
% b=1.7; %0.04,0.03
% e=0.04; % 0.1

for i = 1: num+2
   y = yc(i);
   disp(['y = ',num2str(y),'进度 = ',num2str(i),'/',num2str(num+2)]);
   D1 = img_noise(y:y+lie-1,:);
   %-----去均值   
    Q=zeros(1,row);
 for z=1:row
     temp=D1(:,z);
     Q(z)=mean(temp(temp~=0));
 end
 D = D1-repmat(Q,[size(D1,1),1]);
   D=D/255;
   Dmask = Dm(y:y+lie-1,:);
   omega=find(Dmask~=0);
   [I,J]=ind2sub([lie row],omega);%121--49          不加权效果最好。。。
   [A,E,B]=IALM_reweighted_MR1(D ,I , J, Dictionary2, b/a, e/a , 1e-2, 800,2); %lambda gamma要调试
%    [A,E,B]=test_IALM_sparse(D ,I , J, Dictionary, a,b,e , 1e-7, 800,1);
   output =A*255;  %可以多试试其它方式比较。D-E,或者带Omega的方式
   output = output+repmat(Q,[size(D1,1),1]);   
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

%==========再恢复一次



