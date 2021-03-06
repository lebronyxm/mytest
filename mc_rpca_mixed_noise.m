function [ A,E,Z,iter1 ] = mc_rpca_mixed_noise( D,I,J,tol,maxiter )
%MC_RPCA_MIXED_NOISE implements the inexact augmented lagrange multiplier
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
%   min |A|_* +lambda*|ProjectionOnOmega(E)|_1 + gamma*|ProjectionOnOmega(Z)|_F^2
%   subj A+E+Z=D;
% Copyright:Xinchen YE, Tianjin University
[m,n]=size(D);
if nargin<4
    tol=1e-8;
elseif tol==-1
    tol=1e-8;
end
if nargin<5
    maxiter=200;
elseif maxiter==-1
    maxiter=200;
end

p=length(I);%p is the length of omega,represents the sampling density
rho_s=p/(m*n);%is the sampling density
rho=1.1;%1.1+2.5*rho_s;
lambda =1/sqrt(m);
gamma = 0.1/sqrt(m);
norm_two = lansvd(D, 1, 'L');   %computes the 1 largest singular value
muk=10/norm_two;
d_norm=norm(D,'fro');

Ek=zeros(m,n);Yk=zeros(m,n);
Zk=zeros(m,n);
iter1=0;
converged1=false;

while ~converged1
    iter1 = iter1+1;
    [U,S,V] = svd(D-Ek-Zk+(1/muk)*Yk);
    Ak = U*(shrink(S,1/muk))*V';
    
    Ek = MtOmega(shrink(D-Ak-Zk+(1/muk)*Yk,lambda/muk),I,J,m,n)+...
        D-Ak-Zk+(1/muk)*Yk-MtOmega(D-Ak-Zk+(1/muk)*Yk,I,J,m,n);
    
    Zk = (muk/(muk+2*gamma))*MtOmega(D-Ak-Ek+(1/muk)*Yk,I,J,m,n)+...
        D-Ak-Ek+(1/muk)*Yk-MtOmega(D-Ak-Ek+(1/muk)*Yk,I,J,m,n);
    Yk=Yk+muk*(D-Ak-Ek-Zk);
    muk=rho*muk;
   
     stopCriterion = norm(D-Ak-Ek-Zk, 'fro') / d_norm;
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
Z=Zk;
end

