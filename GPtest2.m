%%
cd /home/hiroyuki/Tone;
clear all;
% Load data .......................
pid=212;
basedir = ['~/Tone/ToneC/',num2str(pid),'/'];
Nf=50; 
wlength=12;  % in points
step=12;       % in points
zeropoint=509;

rejtime=[-0.3 0.5];
[DAT]=loadHGtonedata(pid, 1);

% Trial rejection ....................
% SelT and RejT is the cell array containing trial # for rejefction and
% selection for each channel.
rejperiod = find(DAT.tstamp>rejtime(1) & DAT.tstamp<rejtime(2));
[RejT,SelT] = RJTrial_RRv2(DAT.dat(rejperiod,:,:),3.5);

% Denoise single trials.....................
fprintf('  Denoising ....\n');
[DAT.dat] = denoiseFP(DAT.dat,DAT.fs);
    
% Filtering...................
[DAT, Z, AEP, DAT.rej] = BPfilteringMGv2(DAT,Nf,SelT,RejT,[-0.25 -0.1]);    
[bData,Tind] = timebinV2(Z,wlength,step,zeropoint,DAT.fs); 

% Select timebin.....
F=find(Tind>-0.3 & Tind<0.7); Tind2=Tind(F); 
bData=bData(:,F,:); nt=length(Tind2);

% Accumulate rejected trials........
rejtrial=[];
for n=1:18
    eval(['rejtrial=union(rejtrial,RejT{',num2str(n),'});'])
end
selected=setdiff([1:size(DAT.dat,2)],rejtrial);

% bData is binned data  [ trial - timebin - channel ]
mbData = squeeze(mean(bData(selected,:,:),1));
for n=1:12
    f=find(DAT.seq==n);f=intersect(selected,f);
    eval(['mbData',num2str(n),' = squeeze(mean(bData(f,:,:),1));'])
end    
% mbData = [ Timebin  x  Channel ]
save ~/Tone/GPtest mbData Tind 
clear  rejperiod rejtime Nf zeropoint

%% Make input vector
selch = [1:18];
mbData = mbData(:,selch)';    %   DIMENSION = [ CH x TimeBIN ]
[nch,nt] = size(mbData);
inpt = reshape(mbData, nch*nt,1);
clear F n rejperiod 
%%
close all;
p=3;
% Initialization ====================
[sc, score,eiv,eivec] = PCAanal(mbData); 
% [coef,sc,eigv] = princomp(mbData');
sc=sc(1:p,:)';
x = reshape(sc(:,1:p),nt*p,1);
R=10^-3*eye(nch,nch);
Rbar= blockdiagonal(R,nt);
dL = zeros(nch*nt,1);
C=10^-3*eye(nch,p);
Cbar= blockdiagonal(C,nt);
temp = zeros(p+1,p+1);
temp2 = zeros(nch,p+1);
temp3 = zeros(nch,nch);
temp4 = zeros(nch,p);
param(1:p,1) = 1; % this does not influence anything..
[sc, recon,score,eiv,eivec] = PCAanal(mbData',1);..
param(1:p,2) = 2.0;
param(1:p,3) = .01;   % 0~1
clear sc score eivalue eivector noncomp;
condXmeanOld=norm(x);
close;
%  EM iteration ===================% 
for kk = 1:20
    % E-step ====================% 
    % Constructs covariance matrix .......
    T = 1:length(Tind2);
    for n = 1:p
        [k(:,:,n)] = covFsqEFA(T,T, param(n,:),0);
    end
    K = zeros(p*nt,p*nt);
    for t1 = 1:nt
       for t2 = 1:nt
            indL = (t1-1)*p+1:t1*p;
            indC = (t2-1)*p+1:t2*p;
            tempi = [];
            for ii = 1:p
                tempi = [tempi k(t1,t2,ii)];
            end
            K(indC,indL) = diag(tempi);
       end
    end
    clear tempi ii t1 t2 indL indC 
    % Conditional mean ......
    condXmean = K*Cbar'*inv(Cbar*K*Cbar'+Rbar)*(inpt-dL);
    difX=condXmeanOld-condXmean;
    nd(kk)=norm(difX);
    % Reshape  ( traj = [p x T] )  ......
    traj = reshape(condXmean,p,nt);
    % M-step ======================
    ttj=[];  jjt=[];  
    for n=1:nt
        ttj(:,:,n) = traj(:,n)*traj(:,n)';
    end
    
    for n=1:p
        jjt(:,:,n) = traj(n,:)'*traj(n,:);
    end
    Z = mbData*[traj' ones(size(traj,2),1)];

    for n=1:nt
         temp = temp + [ttj(:,:,n) traj(:,n); traj(:,n)' 1];
         temp2 = temp2 + mbData(:,n)*[traj(:,n)' 1];
    end
    Z = temp2*inv(temp);
    C = Z(:,1:end-1);
    d = C(:,end);

    for n = 1:nt
         temp3 = temp3+ (mbData(:,n)-d)*(mbData(:,n)-d)';
         temp4 = temp4+ (mbData(:,n)-d)*traj(:,n)';
    end
    R = diag(diag(temp3-temp4*C')/nt);  % R= [nch x 1] vector
    
    % Make block diagonal matrix .......
    Rbar= blockdiagonal(R,nt);
    Cbar= blockdiagonal(C,nt);
    dL=repmat(d,nt,1);
    condXmeanOld=condXmean;
    clear  n Z ttj jjt  
end
% Orthonormalization ........
[U,D,V]=svd(C,'econ');
nTraject= D*V'*traj;

%=============================
% figure.................................................
figure; plot(log10(abs(nd)),'.-');
cmp=jet(nt); 

figure;plot3(nTraject(1,:),nTraject(2,:),nTraject(3,:),'-','linewidth',2);
hold on
for n=1:nt
    plot3(nTraject(1,n),nTraject(2,n),nTraject(3,n),'o-', ...
        'markerfacecolor',[cmp(n,1) cmp(n,2) cmp(n,2)],...
        'markeredgecolor','k',...
        'markersize',10);hold on;
end
grid on;   f=find(Tind2>0); f=f(1); f2=find(Tind2>0.2); f2=f2(1);
plot3(nTraject(1,f),nTraject(2,f),nTraject(3,f),'s-', ...
        'markerfacecolor',[cmp(n,1) cmp(n,2) cmp(n,2)],...
        'markeredgecolor','k',...
        'markersize',16);hold on;
plot3(nTraject(1,f2),nTraject(2,f2),nTraject(3,f2),'o-', ...
        'markerfacecolor',[cmp(n,1) cmp(n,2) cmp(n,2)],...
        'markeredgecolor','k',...
        'markersize',16);hold off;
    title('trajectry :  -0.3~0.7sec')
    
    
figure;  plot(Tind2,nTraject');grid on;
%%