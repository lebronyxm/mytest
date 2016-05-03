%inpainting main - seismic
%  clearvars -except Dic_sei
clc;
clear;
load Dic_sei.mat
raw = read_segy_file('C:\Users\Administrator\Desktop\Ӧ����ѧ-�������\680_79_PR.SGY');
img_gray = raw.traces(200:200+399,80:80+399);
img_gray = rot90(img_gray,3); % ��������
[col,row] = size(img_gray);
%%%---------------------------����
lie = 30;
step = 5;
missing = 0.50; 
Lr=1.0; %���еı���
% %%%--------------------------mask =0Ϊȱʧ
[Dm,smask] = creatmask(col,row,missing,Lr);  %����
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
   disp(['y = ',num2str(y),'���� = ',num2str(i),'/',num2str(num+2)]);
   D1 = img_noise(y:y+lie-1,:);
   Dmask = Dm(y:y+lie-1,:);
%-----ȥ��ֵ   
    Q=zeros(1,row);
 for z=1:row
     temp=D1(:,z);
     Q(z)=mean(temp(temp~=0));
 end
 D = D1-repmat(Q,[size(D1,1),1]);
 D = D/10000.*Dmask;  %�������ݲ��ó�255��ͼ����Գ�
%---------   
   omega=find(Dmask~=0);
   [I,J]=ind2sub([lie row],omega);%121--49          ����ȨЧ����á�����
   [A,E,B]=IALM_reweighted_MC(D ,I , J, Dic_sei, 0.1 , 1e-1, 200,3); %һ������1��B��ϡ�裬��С��0.01 ϡ��0.1��700��1e-2
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


