%基于块模式的restoration=缺失+噪声
% clearvars -except Dsort;
clc;

% Pn=ceil(sqrt(160));   %K表示字典的大小,K取324,是平方%初始化一个扁矩阵，bb为字典原子维数开平方，如你要产生256为的，则设置bb=16；
%  bb=9;
%  DCT=zeros(bb,Pn);
%  for k=0:1:Pn-1,     
%     V=cos([0:1:bb-1]'*k*pi/Pn);
%     if k>0
%         V=V-mean(V); 
%     end;
%     DCT(:,k+1)=V/norm(V);
%  end;
%  Dictionary=kron(DCT,DCT)*1;%G就是最终的DCT字典；
% clearvars -except Dic_sei;
% ori_img = double(imread('C:\Users\Administrator\Desktop\test2.jpg'));
ori_img = double(rgb2gray(imread('C:\Users\Administrator\Desktop\fog.jpg')));
ori_img = ori_img(50:450,70:490);

[col, row] = size(ori_img);
%--------parameters
missing = 0.50; 
Lr=0.30; %整行的比例
Dm = creatmask(col,row,missing,Lr);
Dm=ones(col,row);
Dm_ori = Dm;
noise_gs = 20;
noise_sp = 0.20;
% ori_img2 = Add_noise_sparse (ori_img,noise_sp); 
% ori_img2 = Add_noise(ori_img,noise_gs);
% img_noise = ori_img2.*Dm;
img_noise = ori_img;
%比较的时候从下面section开始运行
%%
load Dic_11.mat;
numpatch =20;
blocksize=5;
bs = 2*blocksize+1;
step = 10; %先不用overlapping，直接拼接
p=20; %搜索框
pb=p+blocksize;
ss=col+2*pb;
hh=row+2*pb;
%----------扩展后
Dm=padarray(Dm_ori,[pb pb],'symmetric','pre');  %可以用0扩展，更合理
Dm=padarray(Dm,[pb pb],'symmetric','post');
img_ori=padarray(img_noise,[pb pb],'symmetric','pre');  %可以用0扩展，更合理
img_ori=padarray(img_ori,[pb pb],'symmetric','post');
Pic = img_ori.*Dm;
%---------插值
omega = find(Pic~=0);
[co, ro]=size(Pic);
[I, J]=ind2sub([co ro],omega);
yy = Pic(omega);
[xxi, yyi] = meshgrid(1:ro, 1:co);
img_cz = griddata(J,I,yy,xxi,yyi,'cubic');
img_cz = img_cz(pb+1:end-pb,pb+1:end-pb);
img_ori=padarray(img_cz,[pb pb],'symmetric','pre');  %可以用0扩展，更合理
img_ori=padarray(img_ori,[pb pb],'symmetric','post');
img_sea = img_ori;%  .*Dm; 
%%
%-----------有 step
shu= ceil((col-bs)/step)+1;
heng=ceil((row-bs)/step)+1;
x2=[p+bs:step:(heng-1)*step+p+bs-1,p+row];
y2=[p+bs:step:(shu-1)*step+p+bs-1,p+col];
xyc=zeros(shu*heng,2);
xyc(:,2)=repmat(y2',[heng,1]);
xytemp=repmat(x2,[shu,1]);
xyc(:,1)=xytemp(:);
%% 参数
% a=1.0; %1 250个atoms 10倍DCT的
% b=0.02; % 越大，结构缺失填充越好17.5
% e=0.3; % 0.5 越大细节保留更好  20的噪声用5.2比较好
%----------上面稀疏，下面高斯 0.7-2.7-0.1 没有/255, 归一化：0.9-0.1-0.7
a=1.0; % 
b=0.02; % 越大，2.7  训练字典0.05
e=1.5; %57  第二遍要大一点，防止信息丢失
%%
    backA=zeros(ss,hh); 
    weight=zeros(ss,hh);
%%
for i=1: shu*heng  %
    xc=xyc(i,1);  
    yc=xyc(i,2);
   disp(['x= ',num2str(xc),' y= ',num2str(yc),' 进度= ',num2str(i),'/',num2str(shu*heng)]);    % disp([xc,yc,i,shu*heng]);
    [c_patch,pos,mask,numpatch]=inpainting_block_search(img_sea,xc,yc,blocksize,p,numpatch,Dm);
%     [c_patch,pos,mask]=inpainting_block_search2(img_ref,img_sea,xc,yc,blocksize,p,numpatch,Dm); 
%       img_ref = img_rec;  %还需要扩展
%       img_ref=padarray(img_ref,[pb pb],'symmetric','pre');
%       img_ref=padarray(img_ref,[pb pb],'symmetric','post');
 %---------------
 D1=c_patch.*mask; 
 %----去均值
 Q=zeros(1,numpatch);
 for z=1:numpatch
     temp=D1(:,z);
     Q(z)=mean(temp(temp~=0));
 end
 D = D1-repmat(Q,[size(D1,1),1]);
 D = (D/255).*mask;
 %----检测全0异常
%  if size(unique(D),1)<3;   
%      output2=c_patch;
%  else
omega=find(mask~=0);
[I,J]=ind2sub([bs*bs numpatch],omega);
%  [A,E,B]=test_IALM_sparse(D ,I , J, Dictionary2, a,b,e , 1e-2, 800,2); %1e-7
  [A,E,B]=IALM_reweighted_MR1(D ,I , J, Dictionary2, b/a, e/a , 1e-2, 700,2);
 %---下面两步是限制B的稀疏度，使得相似块用相同的字典
%   B2 = B_sparse( B );
%    A = Dictionary*B2;
output =A.*255; % A,D-E
output2 = output+repmat(Q,[size(D1,1),1]);
% output2 = A+MtOmega(E,I,J,49,numpatch);  %A 或者D-E
%  end
%----------------
  for iter=1:numpatch   %再设计一个加权 之前用15个
    block = col2im(output2(:,iter),[bs bs],[bs bs],'distinct');
    rr = (xc-pb-1+pos(iter,1)):(xc-pb-1+bs-1+pos(iter,1));
    cc = (yc-pb-1+pos(iter,2)):(yc-pb-1+bs-1+pos(iter,2));
    backA(cc, rr)  =  backA(cc, rr) + pos(iter,3) .* block;
    weight(cc, rr) =  weight(cc, rr) + pos(iter,3);
  end
end
img_rec=backA(pb+1:end-pb,pb+1:end-pb)./weight(pb+1:end-pb,pb+1:end-pb);%
normlize = img_rec<0;
img_rec(normlize)=0;
normlize = img_rec>255;
img_rec(normlize)=255;
psnr = calcpsnr(ori_img,img_rec);
ssim = calcssim(img_rec,ori_img);
figure,imshow([uint8(ori_img),uint8(img_rec),uint8(img_noise)]);
title(['ori   &  rec  &  noise  PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim)]);
disp(['PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim)]);

% bayesian_gray;



%------------------不扩展时
% num_row = floor((row-bs)/step); % +1
% num_col = floor((col-bs)/step); % +1
% xc = [blocksize+1:step:num_row*step+blocksize,row-blocksize];
% yc = [blocksize+1:step:num_col*step+blocksize,row-blocksize];
% xyc=zeros((num_row+1)*(num_col+1),2);
% xyc(:,2) = repmat(xc',[num_col+1 1]);
% temp=repmat(yc,[num_row+1,1]);
% xyc(:,1)=temp(:);


%MCA的后半部分，待改成去噪的
% truth = ori_img - img_rec;
% err = img_noise-img_rec;
% Patch_size = 15;
% Num_of_Iterations = 100; %100
% DictSize = 1024;  % 1024
% TextureDict =[];
% CartoonDict =[];
% err2 = err/255;
% % 注意输入的err的数据格式，/255
% [NonRainComponent, RainComponent, Texture_Dict, Cartoon_Dict] = MCA_Image_Decomposition(err2, Patch_size,...
% Num_of_Iterations, DictSize, TextureDict, CartoonDict, 3);
% figure,imshow([abs(truth/255),abs(err2),abs(NonRainComponent),abs(RainComponent)],[]);
% title('与原图err                       err                     非雨                  雨');
% %%
% % img_rec2 = img_rec+RainComponent*255;
% img_rec2 = img_rec+NonRainComponent*255;
% psnr = calcpsnr(ori_img,img_rec2);
% ssim = calcssim(img_rec2,ori_img);
% figure,imshow([uint8(ori_img),uint8(img_rec),uint8(img_noise)]);
% title(['ori   &  rec  &  noise  PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim)]);
% disp(['PSNR = ',num2str(psnr),';  SSIM = ',num2str(ssim)]);
