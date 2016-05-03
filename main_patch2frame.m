%patch2frame�Ȳ���60��60�Ŀ飬���м���2���ӣ���Ȩ4��15���ӡ�
close all;
clear all;
clc;
fid = fopen('hallmonitor.cif', 'r');
row=352;
col=288;
frames=100;%190
video_frame_data = zeros(col,row,3,frames); %�����ǲ�ɫ�ģ��Ҷȼ�ԭmain
 
%�����ô�õĹ��������ԸĳɻҶȵ�֡�ˡ�
for frame=1:frames
    Y=(fread(fid,[row,col]))';
    U=(fread(fid,[row/2,col/2]))';
    V=(fread(fid,[row/2,col/2]))';
     rgb_temp = reshape(yuv420torgb(Y,U,V),288,352,3); %(Y,U,V),288*352,1,3)
     video_frame_data(:,:,:,frame) = rgb_temp;
   % video_frame_data(:,:,frame) = Y;
end
fclose(fid);
%%   %������������
xc=102;yc=154; n=32;  %142,124Ч������
bsk=15;
x2=[xc:bsk:xc+45];y2=[yc:bsk:yc+45];
xyc=zeros(16,2);
xyc(:,2)=repmat(y2',[4,1]);
xytemp=repmat(x2,[4,1]);
xyc(:,1)=xytemp(:);
%%
numpatch=100;
    blocksize=7;
    img_ord = video_frame_data(:,:,:,n); 
    I_3D = video_frame_data(:,:,:,max(1,n-6):min(298,n+6));
    clear video_frame_data;

patchframe=[];%psnratom=zeros(1,36);
param.K=400;  % learns a dictionary with 100 elements
param.lambda=0.15;%0.15
param.numThreads=4; % number of threads
param.batchsize=400;%size of the minibatch
param.mode=5;
param.iter=300; 
psnratom=zeros(1,16);
%load('dictionary.mat')
%for i=1:16
parfor i=1:16 
    xc=xyc(i,1);
    yc=xyc(i,2);
   % disp([xc,yc]);
    c_patch=patch_search_frame(n,img_ord,I_3D,xc,yc,blocksize,numpatch);
    X=c_patch/255; 
    X=X-repmat(mean(X),[size(X,1) 1]);  
    D = mexTrainDL(X,param);
    Dictionary=D;
%%  
 %smask=[35  13 27 5 21 15 29 38 33 48 2 43 7 18 36 31 10 24 41];
 smask=randsample(225,100);
 mask = ones(225,100);
 mask(smask,:)=0;
 test_data=X.*mask; 
 D=test_data; %DΪ���и���ȱʧ��ԭ����
omega=find(D~=0);
[I,J]=ind2sub([225 100],omega);%120Ӧ����100��
[A,E,B]=IALM_reweighted(D ,I , J, Dictionary, 0.1, 1 , 1e-3, 200, c_patch,2);
output = A+MtOmega(E,I,J,225,100); %121--49
Q=c_patch/255;
output2=(output+repmat(mean(Q),[size(Q,1) 1]))*255;
patchatom=output2(:,1);  %ȡ��һ��
%patchatom=mean(output2(:,1:10)')'; %ȡǰʮ�о�ֵ
patchframe=[patchframe patchatom];
psnratom(1,i)=calcpsnr(output2,c_patch);
end
img_rec=col2im(patchframe,[15 15],[60 60],'distinct');
tu=double(rgb2gray(uint8(img_ord(xyc(1,2)-blocksize:xyc(1,2)+52,xyc(1,1)-blocksize:xyc(1,1)+52,:))));
psnr=calcpsnr(tu,img_rec)
figure,subplot(2,1,1);imshow(uint8(tu));title('ԭͼ');
subplot(2,1,2);imshow(uint8(img_rec));title('�ָ�');
%psnratom�е���ÿһ��cpatch��100�еģ���������ֻ���õ�һ�е�ֵ��

