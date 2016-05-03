%inpainting main - single image
%  clearvars -except Dic_sei
clc;
clear;
load Dic_keda_new160.mat;
% load ('F:\programs\matlab\myprogram\Dm.mat');
img_rgb  = imread('G:\picset\test\lena.png'); %G:\picset\test\lena.png
img_gray = double(img_rgb);
[col,row] = size(img_gray);
%%%---------------------------参数
lie = 30;
step = 15;
missing = 0.30; 
Lr=0.3; %整行的比例
% %%%--------------------------mask =0为缺失
Dm = creatmask(col,row,missing,Lr); 
% Dm = ones(col,row);
% Dm(2:2:end,:)=0;
% smask = 2:2:row;
img_noise = img_gray.*Dm;
% img_noise = double(imread('G:\picset\test\lena_2.pgm'));
%%%---------------------------iteration
%%
num = floor((col-lie)/step);
yc = [1:step:num*step+1,col-lie+1]';
backA=zeros(col,row);
weight=zeros(col,row);

for i = 1: num+2
   y = yc(i);
   disp(['y = ',num2str(y),'进度 = ',num2str(i),'/',num2str(num+2)]);
   D1 = img_noise(y:y+lie-1,:);
   Dmask = Dm(y:y+lie-1,:);
%-----去均值   
    Q=zeros(1,row);
 for z=1:row
     temp=D1(:,z);
     Q(z)=mean(temp(temp~=0));
 end
 D = D1-repmat(Q,[size(D1,1),1]);
 D = D/255.*Dmask;  
%---------   
   omega=find(Dmask~=0);
   [I,J]=ind2sub([lie row],omega);%121--49          图像的列模式下，不加权效果最好。。。
   [A,E,B]=IALM_reweighted_MC(D ,I , J, Dictionary, 0.01 , 1e-2, 300,1); % 100次到200次提升0.5dB，300次可能提升不大了
%    [A,E,B]=test_IALM_sparse(D ,I , J, Dictionary, 1,0.001 , 0.7 , 1e-2, 700,1); 
   output =D-E; % A,D-E
   output = output*255;
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
% figure,imshow([uint8(ori_img),uint8(img_rec),uint8(img_noise)],[]);
figure,imshow([img_gray,img_rec,img_noise],[]);
title(['ori   &  rec  &  noise  PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim)]);
disp(['PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim)]);


