%inpainting main - seismic
%  clearvars -except Dic_sei
clc;
clear;
load Dic_sei.mat
raw = read_segy_file('C:\Users\Administrator\Desktop\应用数学-数据填充\680_79_PR.SGY');
img_gray = raw.traces(200:200+399,80:80+399);
img_gray = rot90(img_gray,3); % 地震数据
[col,row] = size(img_gray);
%%%---------------------------参数
lie = 30;
step = 5;
missing = 0.50; 
Lr=1.0; %整行的比例
% %%%--------------------------mask =0为缺失
[Dm,smask] = creatmask(col,row,missing,Lr);  %地震
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
 D = D/10000.*Dmask;  %地震数据不用除255，图像可以除
%---------   
   omega=find(Dmask~=0);
   [I,J]=ind2sub([lie row],omega);%121--49          不加权效果最好。。。
   [A,E,B]=IALM_reweighted_MC(D ,I , J, Dic_sei, 0.1 , 1e-1, 200,3); %一般设置1，B不稀疏，调小到0.01 稀疏0.1，700次1e-2
   output =D-E; % A,D-E
   output = output*10000;
   output = output+repmat(Q,[size(D1,1),1]);
   backA(y:y+lie-1,:) = backA(y:y+lie-1,:)+output;
   weight(y:y+lie-1,:) = weight(y:y+lie-1,:)+1;
end
img_rec = backA./weight;
img_gray = rot90(img_gray);
img_noise = rot90(img_noise);
img_rec = rot90(img_rec);
psnr = calcpsnr(img_gray,img_rec);
ssim = calcssim(img_rec,img_gray);
snr = calcsnr(img_gray,img_rec);
% figure,imshow([img_gray,img_rec,img_noise],[]);
% title(['ori   &  rec  &  noise  PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim),'  SNR = ',num2str(snr)]);
disp(['PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim),' SNR =',num2str(snr)]);

% figure,subplot(131);
% wigb(img_gray);title('original data');xlabel('offset');ylabel('time samples')
% subplot(132);
% wigb(img_noise,0.4);title('missing traces section');
% xlabel('offset');
% subplot(133);
% %wigb(INTDATA(:,min(h):max(h)));
% wigb(img_rec);title(['rec  SNR = ',num2str(snr)]);
% xlabel('offset');
s_compare(img_gray,img_rec);title(['ours SNR = ',num2str(snr)])


