% AllAlgorithm_Experiment_full_SH_tuned.m
% 针对 data_SH 的重建增强版本：
% 1) 分窗 GRL（早窗/晚窗分别重建并融合），平衡 H/S 强度差异
% 2) 深度补偿（按时间 bin 增益）+ 鲁棒归一化，抑制强目标淹没弱目标
% 3) 输出方向校正（rot90/flip）使结果与实验场景一致

clc; clear;

%% 路径
root_code = './';
root_data = './data';
addpath(root_code); addpath(root_data);

%% 载入数据
S = load(fullfile(root_data, 'data_SH.mat'));
if isfield(S,'data'), data_raw = S.data; else, error('missing data'); end

data0 = max(double(permute(data_raw,[3,2,1])),0); % [T,H,W]
[T0,H0,W0] = size(data0);

%% IRF
S_irf = load(fullfile(root_data,'irf.mat'));
if isfield(S_irf,'IRF'), IRF = S_irf.IRF; else, IRF = S_irf.irf; end
Tcrop = 1024;
h = zeros(Tcrop,1);
h(1:min(numel(IRF),Tcrop)) = IRF(:);
h = medfilt1(h,5);
h = h / max(sum(h),eps);
[~,shift_t] = max(h);
h = [h(shift_t:end); zeros(shift_t-1,1)];

%% first-bounce 对齐
[~,index] = max(data0,[],1);
tofgrid = squeeze(index);
data_shift = zeros(size(data0));
for iy=1:H0
    for ix=1:W0
        t0 = tofgrid(iy,ix);
        data_shift(:,iy,ix) = circshift(data0(:,iy,ix), -min(T0,t0));
    end
end
data_shift(1:min(140,size(data_shift,1)),:,:) = 0;
data = data_shift(1:min(Tcrop,size(data_shift,1)),:,:);
[T,H,W] = size(data);
sz = size(data);

%% SH 推荐参数
width = 1.12; range = 0.7; sigma_psf = 2.3;
myeps = 0.02;

% 双窗：针对 H(近) 和 S(远)
win1 = [150,220];   % 早窗（通常更强）
win2 = [320,420];   % 晚窗（通常更弱）

iters1 = 35;        % 强目标少迭代，避免过饱和
iters2 = 60;        % 弱目标多迭代，提细节
lambda_fuse = 0.55; % 晚窗权重（可 0.45~0.65 调）

%% 构造算子
psf = definePsf(sz(2), sz(1), width./range, sigma_psf);
fpsf = fftn(psf);
[mtx,mtxi] = resamplingOperator(sz(1));
mtx = full(mtx); mtxi = full(mtxi);
cut = @(x) x(1:sz(1),1:sz(2),1:sz(3));
flatten32 = @(x) x(:,:);

H_mat = zeros([sz(1),sz(1)]);
for i = 1:sz(1), H_mat(i,:) = circshift(h,i-1); end

A_H = @(x) real( reshape( ...
    H_mat * (mtxi * flatten32(cut(ifftn(fftn(padarray(reshape(mtx*x(:,:),sz),sz,0,'post')).*fpsf)))) ...
    , sz));
At_H = @(x) real( reshape( ...
    H_mat' * (mtxi * flatten32(cut(ifftn(fftn(padarray(reshape(mtx*x(:,:),sz),sz,0,'post')).*conj(fpsf))))) ...
    , sz));

%% 深度补偿（仅用于 RL 观测，不改前向模型）
t = (1:T)';
depth_gain = (t / max(t)).^1.2;   % 1.0~1.5 可调
D = reshape(depth_gain,[T,1,1]);
data_bal = data .* D;
data_bal = data_bal / (prctile(data_bal(:),99.7)+eps);

%% 分窗 GRL 重建
vol1 = run_grl_window(data_bal, A_H, At_H, win1, iters1, 0.6);
vol2 = run_grl_window(data_bal, A_H, At_H, win2, iters2, 0.45);

% 分窗结果鲁棒归一化后融合，避免强窗压制弱窗
v1 = normalize_robust(sum(vol1(win1(1):win1(2),:,:),1));
v2 = normalize_robust(sum(vol2(win2(1):win2(2),:,:),1));
P_sh = (1-lambda_fuse).*v1 + lambda_fuse.*v2;
P_sh = P_sh / (max(P_sh(:))+eps);

%% 方向校正（根据你的场景选择）
% 常用：横过来 -> 旋转 + 翻转
P_sh_oriented = rot90(P_sh, -1);
P_sh_oriented = fliplr(P_sh_oriented);

%% 显示
figure('Color','w');
subplot(1,3,1); imagesc(squeeze(v1)); axis image off; colormap turbo; title('Early window (H)');
subplot(1,3,2); imagesc(squeeze(v2)); axis image off; colormap turbo; title('Late window (S)');
subplot(1,3,3); imagesc(P_sh_oriented); axis image off; colormap turbo; title('Fused + oriented SH');

assignin('base','P_sh',P_sh);
assignin('base','P_sh_oriented',P_sh_oriented);

%% ===== local =====
function vol = run_grl_window(data_in, A_H, At_H, win, nIter, smoothSigma)
    sz = size(data_in);
    td1 = max(1,win(1)); td2 = min(sz(1),win(2));
    mask = false(sz); mask(td1:td2,:,:) = true;
    M = double(mask);

    input = data_in .* M;
    est = max(At_H(input), 1e-8);
    est = est/(max(est(:))+eps) * (max(input(:))+eps);

    sens = max(At_H(M), 1e-8);

    for k=1:nIter
        pred = A_H(est) + 1e-9;
        ratio = zeros(sz);
        ratio(mask) = input(mask) ./ pred(mask);
        back = At_H(ratio);
        est = max(est .* back ./ sens, 0);

        if mod(k,2)==0
            est = smooth3(est,'gaussian',[1,3,3],smoothSigma);
        end
    end
    vol = est;
end

function out = normalize_robust(x)
    x = squeeze(x);
    x = x - prctile(x(:),2);
    x = max(x,0);
    x = x / (prctile(x(:),99.5)+eps);
    out = min(max(x,0),1);
end

function psf = definePsf(U,V,slope,sigma)
    x = linspace(-1,1,2.*U);
    y = linspace(-1,1,2.*U);
    z = linspace(0,2,2.*V);
    [grid_z,grid_y,grid_x] = ndgrid(z,y,x);
    psf = abs(((4.*slope).^2).*(grid_x.^2 + grid_y.^2) - grid_z);
    psf = double(psf == repmat(min(psf,[],1),[2.*V 1 1]));
    psf = reshape(fspecial('gaussian',[size(psf,2),size(psf,3)],sigma),[1,size(psf,2),size(psf,3)]).*psf;
    psf = psf./sum(psf(:,U,U));
    psf = psf./norm(psf(:));
    psf = circshift(psf,[0 U U]);
end

function [mtx,mtxi] = resamplingOperator(M)
    mtx = sparse([],[],[],M.^2,M,M.^2);
    x = 1:M.^2;
    mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
    mtx  = spdiags(1./sqrt(x)',0,M.^2,M.^2)*mtx;
    mtxi = mtx';
    K = log(M)./log(2);
    for k = 1:round(K)
        mtx  = 0.5.*(mtx(1:2:end,:)  + mtx(2:2:end,:));
        mtxi = 0.5.*(mtxi(:,1:2:end) + mtxi(:,2:2:end));
    end
end
