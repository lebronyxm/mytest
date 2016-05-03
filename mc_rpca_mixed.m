function [ A,E,iter1 ] = mc_rpca_mixed( D,I,J,tol,maxiter )
%MC_RPCA_MIXED implements the inexact augmented lagrange multiplier
% method for matrix recovery with erase and sparse noise
%   D-m*n matrix of observations
%
% lambda-weight on sparse error term in the cost function
%
% tol-tolerance for stopping criterion
%    -DEFAULT 1e-7 if omitted or -1
%
% maxiter-maximum number of iterations
%    -DEFAULT 1000 if omitted or -1
%
% Model:
%   min |A|_* +lambda*|ProjectionOnOmega(E)|_1
%   subj A+E=D;
% Copyright:Xinchen YE, Tianjin University

[m,n]=size(D);
if nargin<4
    tol=1e-10;
elseif tol==-1
    tol=1e-10;
end
if nargin<5
    maxiter=200;
elseif maxiter==-1
    maxiter=200;
end

p=length(I);%p is the length of omega,represents the sampling density
rho_s=p/(m*n);%is the sampling density
rho=1.1;%1.1+2.5*rho_s;
lambda =0.08;%1/sqrt(m);
norm_two = lansvd(D, 1, 'L');   %computes the 1 largest singular value
muk=10/norm_two;
d_norm=norm(D,'fro');

Ek=zeros(m,n);Yk=zeros(m,n);
iter1=0;
converged1=false;

while ~converged1wwww
    iter1 = iter1+1;
    [U,S,V] = svd(D-Ek+(1/muk)*Yk);
    Ak = U*(shrink(S,1/muk))*V';
   %这个E的和大师兄的一样 
    Ek = MtOmega(shrink(D-Ak+(1/muk)*Yk,lambda/muk),I,J,m,n)+...
        D-Ak+(1/muk)*Yk-MtOmega(D-Ak+(1/muk)*Yk,I,J,m,n);
    
    Yk=Yk+muk*(D-Ak-Ek);
    muk=rho*muk;
   
     stopCriterion = norm(D-Ak-Ek, 'fro') / d_norm;
      disp([ ' r(F) ' num2str(rank(Ak))...
            ' |E|_0 ' num2str(length(find(abs(Ek)>0)))...
            ' stopCriterion ' num2str(stopCriterion)  ' iter1 ' num2str(iter1) ' mu ' num2str(muk)]);
    if stopCriterion < tol
        converged1 = true;
    end   
    if ~converged1&&iter1>=maxiter
        disp('Maximum iterations reached');
        converged1=true;
    end
end

A=Ak;
E=Ek;
end

