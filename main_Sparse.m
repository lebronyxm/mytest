% 高斯的块的最终主函数
%基于块模式的restoration=缺失+高斯噪声

clc;
% clearvars -except Dic_sei;
ori_img = double(imread('G:\picset\test\26.gif'));
% ori_img = double(rgb2gray(imread('G:\picset\test\pepper_gray.png')));
% ori_img = ori_img(200:400,230:420);
[col, row] = size(ori_img);
%--------parameters
% missing = 0.10; 
% Lr=0.30; %整行的比例
% Dm = creatmask(col,row,missing,Lr);
Dm_ori = Dm;
noise_sp = 0.40;
ori_img2 = Add_noise_sparse (ori_img,noise_sp); 
img_noise = ori_img2.*Dm;
%%
load Dic_11.mat;
numpatch =20;
blocksize=5;
bs = 2*blocksize+1;
step = bs; %先不用overlapping，直接拼接
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
%%------------
shu= ceil((col-bs)/step)+1;
heng=ceil((row-bs)/step)+1;
x2=[p+bs:step:(heng-1)*step+p+bs-1,p+row];
y2=[p+bs:step:(shu-1)*step+p+bs-1,p+col];
xyc=zeros(shu*heng,2);
xyc(:,2)=repmat(y2',[heng,1]);
xytemp=repmat(x2,[shu,1]);
xyc(:,1)=xytemp(:);
%%----------- 参数
a=1.0; %1 250个atoms 10倍DCT的
b=0.02; % 越大，结构缺失填充越好17.5
e= 0.4 - 0.5*noise_sp ; % 0.5 越大细节保留更好 0.3--20  
%%-----------
    backA=zeros(ss,hh); 
    weight=zeros(ss,hh);
for i=1: shu*heng  %
    xc=xyc(i,1);  
    yc=xyc(i,2);
   disp(['x= ',num2str(xc),' y= ',num2str(yc),' 进度= ',num2str(i),'/',num2str(shu*heng)]);    % disp([xc,yc,i,shu*heng]);
    [c_patch,pos,mask,numpatch]=inpainting_block_search(img_sea,xc,yc,blocksize,p,numpatch,Dm);
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
 [A,E,B]=test_IALM_sparse(D ,I , J, Dictionary2, a,b,e , 1e-2, 700,1); %1e-7
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
%%%%%%%----------------------------------------d第二次
numpatch =20;
step = 6;
img_ref = img_rec;  %还需要扩展
img_ref=padarray(img_ref,[pb pb],'symmetric','pre');
img_ref=padarray(img_ref,[pb pb],'symmetric','post');
shu= ceil((col-bs)/step)+1;
heng=ceil((row-bs)/step)+1;
x2=[p+bs:step:(heng-1)*step+p+bs-1,p+row];
y2=[p+bs:step:(shu-1)*step+p+bs-1,p+col];
xyc=zeros(shu*heng,2);
xyc(:,2)=repmat(y2',[heng,1]);
xytemp=repmat(x2,[shu,1]);
xyc(:,1)=xytemp(:);
backA=zeros(ss,hh);
weight=zeros(ss,hh);
for i=1: shu*heng  %
    xc=xyc(i,1);  
    yc=xyc(i,2);
   disp(['x= ',num2str(xc),' y= ',num2str(yc),' 进度= ',num2str(i),'/',num2str(shu*heng)]);    % disp([xc,yc,i,shu*heng]);
    [c_patch,pos,mask]=inpainting_block_search2(img_ref,img_sea,xc,yc,blocksize,p,numpatch,Dm); 
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
 [A,E,B]=test_IALM_sparse(D ,I , J, Dictionary2, a,b,e , 1e-2, 700,2); %1e-7
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


