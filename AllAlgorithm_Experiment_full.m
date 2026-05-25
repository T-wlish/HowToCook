% AllAlgorithm_Experiment_full.m
% =============================================================
% 全算法对比（实验数据）
%  1) GRL（含实测 IRF: H）
%  2) Reg-Inv（invA_H）
%  3) RL(1D)+Solve（先时域RL，再 invA_I）
%  4) FBP / LCT / f-k / Phasor
% =============================================================

clc; clear;

%% 路径
root_code = 'D:\AAAA--Very Important_2025\20240304 Experiment\LucyRichardson from XWH\Code\MoreCompareAlgorithm\CH code';
root_data = 'D:\AAAA--Very Important_2025\20240304 Experiment\LucyRichardson from XWH\Code\data';
addpath(root_code); addpath(root_data);
fprintf('Code path: %s\n', root_code);
fprintf('Data path: %s\n', root_data);

%% 1) 选择数据
c = menu('选择实验数据', 'data_CH.mat', 'data_CwithTitledH.mat', 'data_UCAS.mat', 'data_SH.mat');
if c==0, return; end
switch c
    case 1, data_file = 'data_CH.mat';
    case 2, data_file = 'data_CwithTitledH.mat';
    case 3, data_file = 'data_UCAS.mat';
    case 4, data_file = 'data_SH.mat';
end
fprintf('[Data] %s\n', fullfile(root_data, data_file));

%% 2) 载入
S = load(fullfile(root_data, data_file));
if isfield(S,'data')
    data_raw = S.data;
else
    fn = fieldnames(S); pick='';
    for k=1:numel(fn)
        if isnumeric(S.(fn{k})) && ndims(S.(fn{k}))==3, pick=fn{k}; break; end
    end
    if isempty(pick), error('未找到3D变量'); end
    data_raw = S.(pick);
end

data0 = max(double(permute(data_raw,[3,2,1])),0); % [T,H,W]
[T0,H0,W0] = size(data0);
fprintf('[Load] data0 [T,H,W]=[%d,%d,%d]\n', T0,H0,W0);

%% 3) IRF
S_irf = load(fullfile(root_data,'irf.mat'));
if isfield(S_irf,'IRF'), IRF=S_irf.IRF; elseif isfield(S_irf,'irf'), IRF=S_irf.irf; else, error('irf missing'); end
Tcrop = 1024;
h = zeros(Tcrop,1); h(1:min(numel(IRF),Tcrop)) = IRF(:);
h = medfilt1(h,5); h = h/max(sum(h),eps);
[~,st] = max(h); h = [h(st:end); zeros(st-1,1)];

%% 4) first-bounce 对齐 + 裁剪
[~,idx] = max(data0,[],1);
tofgrid = squeeze(idx);
data_shift = zeros(size(data0));
for iy=1:H0
    for ix=1:W0
        t0 = min(T0,tofgrid(iy,ix));
        data_shift(:,iy,ix) = circshift(data0(:,iy,ix), -t0);
    end
end
data_shift(1:min(140,size(data_shift,1)),:,:) = 0;
data = data_shift(1:min(Tcrop,size(data_shift,1)),:,:);
[T,H,W] = size(data); sz = size(data);
fprintf('[Align+Crop] data [T,H,W]=[%d,%d,%d]\n', T,H,W);

%% 5) 构造算子
width = 1.12; range = 0.7; sigma_psf = 2.5;
if contains(lower(data_file),'ch'), sigma_psf = 1;
elseif contains(lower(data_file),'titledh'), sigma_psf=2;
elseif contains(lower(data_file),'ucas'), sigma_psf=2.5;
end
myeps = 0.01;

H_mat = zeros(T,T);
for i=1:T, H_mat(i,:) = circshift(h(1:T), i-1); end

psf = definePsf(sz(2), sz(1), width./range, sigma_psf);
fpsf = fftn(psf);
[mtx, mtxi] = resamplingOperator(sz(1));
mtx = full(mtx); mtxi = full(mtxi);
cut = @(x) x(1:sz(1),1:sz(2),1:sz(3));
flatten32 = @(x) x(:,:);

A_H = @(x) real(reshape( ...
    H_mat * (mtxi * flatten32( ...
    cut(ifftn(fftn(padarray(reshape(mtx*x(:,:),sz),sz,0,'post')).*fpsf)) )) ...
    , sz));

At_H = @(x) real(reshape( ...
    H_mat' * (mtxi * flatten32( ...
    cut(ifftn(fftn(padarray(reshape(mtx*x(:,:),sz),sz,0,'post')).*conj(fpsf))) )) ...
    , sz));

invA_H = @(x,lambda) max(real(reshape( ...
    pinv(H_mat + myeps*eye(T)) * (mtxi * flatten32( ...
    cut(ifftn(fftn(padarray(reshape(mtx*x(:,:),sz),sz,0,'post')).*(conj(fpsf)./(fpsf.*conj(fpsf)+lambda)))) )) ...
    , sz)),0);

I_T = speye(T);
invA_I = @(x,lambda) max(real(reshape( ...
    pinv(I_T + myeps*eye(T)) * (mtxi * flatten32( ...
    cut(ifftn(fftn(padarray(reshape(mtx*x(:,:),sz),sz,0,'post')).*(conj(fpsf)./(fpsf.*conj(fpsf)+lambda)))) )) ...
    , sz)),0);

%% 6) 参数
params = struct();
if contains(lower(data_file),'ch')
    params.margin_InvA1=200; params.margin_InvA2=300; params.margin_GRL1=40; params.margin_GRL2=320;
elseif contains(lower(data_file),'ucas')
    params.margin_InvA1=280; params.margin_InvA2=310; params.margin_GRL1=230; params.margin_GRL2=310;
elseif contains(lower(data_file),'titledh')
    params.margin_InvA1=280; params.margin_InvA2=330; params.margin_GRL1=210; params.margin_GRL2=310;
else
    params.margin_InvA1=250; params.margin_InvA2=430;
    params.use_sh_auto_window = true;
    params.sh_search_range = [120,360];
    params.sh_td1_min = 120; params.sh_td2_max = 380;
    params.sh_guard_after_firstbounce = 150;
    params.sh_auto_half_pre = 70; params.sh_auto_half_post = 80;
    params.sh_mass_q1=0.18; params.sh_mass_q2=0.88;
    params.sh_twopeak_enable = true; params.sh_twopeak_ratio = 0.40; params.sh_twopeak_gapmin = 26;
    params.sh_win_min_len = 150; params.sh_win_max_len = 280;

    params.sh_depth_gain_exp = 0.78;
    params.sh_second_pass_enable = true;
    params.sh_second_pass_pre = 65; params.sh_second_pass_post = 95; params.sh_second_pass_min_gap = 22;
    params.sh_volume_aux_weight = 0.50;

    params.sh_project_mode = 'hybrid';
    params.sh_vertical_balance_enable = true;
    params.sh_target_top_bottom_ratio = 0.82;
    params.sh_balance_max_boost = 1.55;
    params.sh_second_pass_blend = 0.74;
    params.sh_top_boost = 0.26;
    params.sh_sum_weight = 0.82;

    params.sh_enable_grid_search = true;
    params.sh_enable_secondpass_grid = true;
    params.margin_GRL1 = 140; params.margin_GRL2 = 330;
end
params.lambda_InvA = 10; params.lambda_RLsolve = 0.5; params.iters_GRL = 50; params.iters_RL1D = 8;
td1 = max(1,params.margin_GRL1); td2 = min(T,params.margin_GRL2);
fprintf('[Params] window=[%d,%d], InvA=[%d,%d], GRL=[%d,%d]\n', td1,td2,params.margin_InvA1,params.margin_InvA2,params.margin_GRL1,params.margin_GRL2);

%% 7) Reg-Inv
input_inva = data;
input_inva(1:params.margin_InvA1,:,:) = 0;
input_inva(params.margin_InvA2:end,:,:) = 0;
input_inva(isnan(input_inva))=0;
img_inva = invA_H(input_inva, params.lambda_InvA);

%% 8) GRL
is_sh = contains(lower(data_file),'sh');
estimatedImage2 = []; has_second_pass = false; td1b = td1; td2b = td2; pk_aux = round((td1+td2)/2);
if is_sh
    if params.use_sh_auto_window
        tprof = movmean(squeeze(sum(sum(data,2),3)),9);
        s1 = max(1,params.sh_search_range(1)); s2 = min(T,params.sh_search_range(2));
        e = max(tprof(s1:s2),0); ce = cumsum(e)/(sum(e)+eps);
        i1 = find(ce>=params.sh_mass_q1,1,'first'); if isempty(i1), i1=1; end
        i2 = find(ce>=params.sh_mass_q2,1,'first'); if isempty(i2), i2=numel(e); end
        td1_mass = s1+i1-1; td2_mass = s1+i2-1;

        ratio = zeros(T,1);
        for tt=s1:s2
            pre=max(1,tt-18):tt-1; post=tt+1:min(T,tt+18);
            if isempty(pre)||isempty(post), continue; end
            ratio(tt)=tprof(tt)/(0.5*mean(tprof(pre))+0.5*mean(tprof(post))+eps);
        end
        first_guard = max(params.sh_td1_min, min(T-5, params.sh_guard_after_firstbounce));
        ratio(1:first_guard)=0;
        [~,pk_main]=max(ratio);
        if pk_main<=first_guard || ~isfinite(pk_main)
            [~,rpk]=max(tprof(s1:s2)); pk_main=s1+rpk-1;
        end
        pk_aux = pk_main;
        if params.sh_twopeak_enable
            ratio2 = ratio;
            a=max(s1,pk_main-params.sh_twopeak_gapmin); b=min(s2,pk_main+params.sh_twopeak_gapmin);
            ratio2(a:b)=0;
            [v2,p2]=max(ratio2);
            if v2 > params.sh_twopeak_ratio*ratio(pk_main), pk_aux=p2; end
        end
        td1_peak = min(pk_main,pk_aux)-params.sh_auto_half_pre;
        td2_peak = max(pk_main,pk_aux)+params.sh_auto_half_post;
        td1_peak = min(td1_peak, min(pk_main,pk_aux)-45);
        td2_peak = max(td2_peak, max(pk_main,pk_aux)+55);

        td1 = max(params.sh_td1_min, round(0.6*td1_mass + 0.4*td1_peak));
        td2 = min(params.sh_td2_max, round(0.6*td2_mass + 0.4*td2_peak));

        wlen = td2-td1+1;
        if wlen < params.sh_win_min_len
            need = params.sh_win_min_len - wlen;
            td1 = max(params.sh_td1_min, td1-floor(need/2));
            td2 = min(params.sh_td2_max, td2+ceil(need/2));
        elseif wlen > params.sh_win_max_len
            cutn = wlen-params.sh_win_max_len;
            td1 = td1+floor(cutn/2); td2 = td2-ceil(cutn/2);
        end
    end

    fprintf('[GRL-SH] selected window=[%d,%d], len=%d\n', td1,td2,td2-td1+1);
    fprintf('[GRL-SH] peaks main/aux=[%d,%d]\n', round((td1+td2)/2), pk_aux);

    mask = false(sz); mask(td1:td2,:,:) = true; M = double(mask);
    input_grl = data .* M; input_grl(isnan(input_grl))=0;
    dg = reshape(((1:T)'/T).^params.sh_depth_gain_exp,[T,1,1]);
    input_grl = input_grl .* dg;
    bg_t = median(reshape(input_grl,T,[]),2);
    input_grl = max(input_grl - reshape(bg_t,[T,1,1]),0);

    estimatedImage = max(At_H(input_grl),1e-8);
    estimatedImage = estimatedImage/(max(estimatedImage(:))+eps) * (max(input_grl(:))+eps);
    sens = max(At_H(M),1e-8);

    sh_iters = 40;
    fprintf('[GRL-SH] auto window=[%d,%d], iters=%d ...\n', td1,td2,sh_iters);
    for k=1:sh_iters
        convImage = A_H(estimatedImage)+1e-9;
        ratio = zeros(sz); ratio(mask)=input_grl(mask)./convImage(mask);
        backConv = At_H(ratio);
        estimatedImage = max(estimatedImage .* backConv ./ sens, 0);
        estimatedImage(~isfinite(estimatedImage))=0;
        if mod(k,3)==0, estimatedImage = smooth3(estimatedImage,'gaussian',[1,3,3],0.45); end
    end

    if params.sh_second_pass_enable && abs(pk_aux-round((td1+td2)/2)) >= params.sh_second_pass_min_gap
        td1b = max(params.sh_td1_min, pk_aux-params.sh_second_pass_pre);
        td2b = min(params.sh_td2_max, pk_aux+params.sh_second_pass_post);
        mask2 = false(sz); mask2(td1b:td2b,:,:) = true; M2 = double(mask2);
        in2 = data .* M2; in2(isnan(in2))=0;
        bg2 = median(reshape(in2,T,[]),2); in2 = max(in2-reshape(bg2,[T,1,1]),0);
        estimatedImage2 = max(At_H(in2),1e-8);
        estimatedImage2 = estimatedImage2/(max(estimatedImage2(:))+eps) * (max(in2(:))+eps);
        sens2 = max(At_H(M2),1e-8);
        for k=1:round(0.95*sh_iters)
            conv2 = A_H(estimatedImage2)+1e-9;
            ratio2 = zeros(sz); ratio2(mask2)=in2(mask2)./conv2(mask2);
            back2 = At_H(ratio2);
            estimatedImage2 = max(estimatedImage2 .* back2 ./ sens2,0);
            estimatedImage2(~isfinite(estimatedImage2))=0;
            if mod(k,3)==0, estimatedImage2 = smooth3(estimatedImage2,'gaussian',[1,3,3],0.45); end
        end
        estimatedImage = (1-params.sh_volume_aux_weight)*estimatedImage + params.sh_volume_aux_weight*estimatedImage2;
        has_second_pass = true;
        fprintf('[GRL-SH] second-pass enabled, aux window=[%d,%d]\n', td1b, td2b);
    end

    % 残差驱动第三支路：提取主重建未解释的高时延成分（常对应上方S）
    if isfield(params,'sh_residual_pass_enable') && params.sh_residual_pass_enable
        pred_main = A_H(estimatedImage);
        resid = max(input_grl - pred_main, 0);
        late1 = min(T, max(td1, round(td1 + 0.45*(td2-td1))));
        late2 = td2;
        mask3 = false(sz); mask3(late1:late2,:,:) = true;
        resid = resid .* mask3;
        if nnz(resid) > 0
            est3 = max(At_H(resid),1e-8);
            est3 = est3/(max(est3(:))+eps) * (max(resid(:))+eps);
            sens3 = max(At_H(double(mask3)),1e-8);
            for k=1:round(0.55*sh_iters)
                conv3 = A_H(est3)+1e-9;
                r3 = zeros(sz); r3(mask3)=resid(mask3)./conv3(mask3);
                est3 = max(est3 .* At_H(r3) ./ sens3, 0);
                if mod(k,3)==0, est3 = smooth3(est3,'gaussian',[1,3,3],0.45); end
            end
            estimatedImage = 0.80*estimatedImage + 0.20*est3;
            fprintf('[GRL-SH] residual-pass enabled, late window=[%d,%d]\n', late1, late2);
        end
    end

    P_sum = squeeze(sum(estimatedImage(td1:td2,:,:),1));
    P_max = squeeze(max(estimatedImage(td1:td2,:,:),[],1));

    if strcmpi(params.sh_project_mode,'hybrid')
        if has_second_pass && ~isempty(estimatedImage2)
            P_aux = squeeze(sum(estimatedImage2(td1b:td2b,:,:),1));
            P_aux = P_aux/(prctile(P_aux(:),99.5)+eps);
            P_sum = P_sum/(prctile(P_sum(:),99.5)+eps);
            P_sum = (1-params.sh_second_pass_blend)*P_sum + params.sh_second_pass_blend*P_aux;
            [H2,W2] = size(P_sum);
            topMask = zeros(H2,1); topMask(1:round(0.55*H2))=1;
            P_sum = P_sum + params.sh_top_boost*(topMask*ones(1,W2)).*P_aux;
        end
        P_sum = P_sum/(prctile(P_sum(:),99.7)+eps);
        P_max = P_max/(prctile(P_max(:),99.7)+eps);
        P_grl = params.sh_sum_weight*P_sum + (1-params.sh_sum_weight)*P_max;
    else
        P_grl = P_sum;
    end
    P_grl = P_grl/(prctile(P_grl(:),99.7)+eps); P_grl = min(max(P_grl,0),1);

    % 自适应上下能量平衡：抑制下方过强、提升上方弱回波
    if isfield(params,'sh_vertical_balance_enable') && params.sh_vertical_balance_enable
        [HH,WW] = size(P_grl);
        split = max(2, min(HH-1, round(0.55*HH)));
        Et = sum(sum(P_grl(1:split,:)));
        Eb = sum(sum(P_grl(split+1:end,:))) + eps;
        r = Et / Eb;
        target = params.sh_target_top_bottom_ratio;
        if r < target
            g = min(params.sh_balance_max_boost, target / max(r,1e-6));
            topMask = zeros(HH,1);
            topMask(1:split) = linspace(1,0.35,split);
            botMask = zeros(HH,1);
            botMask(split+1:end) = linspace(0.12,0,numel(split+1:HH));
            P_grl = P_grl .* (1 + (g-1)*(topMask*ones(1,WW))) .* (1 - 0.15*(botMask*ones(1,WW)));
            P_grl = P_grl/(prctile(P_grl(:),99.7)+eps);
            P_grl = min(max(P_grl,0),1);
            fprintf('[GRL-SH] vertical-balance applied: ratio %.3f -> target %.3f, gain=%.3f\n', r, target, g);
        else
            fprintf('[GRL-SH] vertical-balance skipped: ratio %.3f >= target %.3f\n', r, target);
        end
    end
else
    mask = false(sz); mask(td1:td2,:,:) = true; M=double(mask);
    input_grl = data.*M; input_grl(isnan(input_grl))=0;
    estimatedImage = max(At_H(input_grl),1e-8);
    estimatedImage = estimatedImage/(max(estimatedImage(:))+eps)*(max(input_grl(:))+eps);
    sens = max(At_H(M),1e-8);
    for k=1:params.iters_GRL
        convImage = A_H(estimatedImage)+1e-9;
        ratio = zeros(sz); ratio(mask)=input_grl(mask)./convImage(mask);
        estimatedImage = max(estimatedImage .* At_H(ratio) ./ sens,0);
        if mod(k,2)==0, estimatedImage = smooth3(estimatedImage,'gaussian',[1,3,3],0.5); end
    end
    P_grl = squeeze(sum(estimatedImage(td1:td2,:,:),1));
    P_grl = P_grl/(max(P_grl(:))+eps);
end

%% 9) RL(1D)+Solve
data_rl_in = data;
data_rl_in(1:params.margin_GRL1,:,:) = 0;
data_rl_in(params.margin_GRL2:end,:,:) = 0;
data_rl_in(isnan(data_rl_in)) = 0;
optsR = struct('nIter',params.iters_RL1D,'deconv_range',[td1,td2],'keep_full_length',true,'eps',1e-6*max(1,mean(data_rl_in(:))));
data_rl1d = rl_deconv_timehist_fixedpsf_window(data_rl_in, h, optsR);
blurred_rl = data_rl1d/(max(data_rl1d(:))+eps);
img_rl1dsolve = invA_I(blurred_rl, params.lambda_RLsolve);

%% 10) CNLOS
meas_hwT = max(permute(data0,[2,3,1]),0);
[~,tofgrid_raw] = max(meas_hwT,[],3);
P_fbp=[]; P_lct=[]; P_fk=[]; P_phasor=[];
try, v=EFsimulation_cnlos_reconstruction(meas_hwT,tofgrid_raw,width,0); P_fbp=max(squeeze(max(v,[],3)),0); P_fbp=P_fbp/(max(P_fbp(:))+eps); catch, end
try, v=EFsimulation_cnlos_reconstruction(meas_hwT,tofgrid_raw,width,1); P_lct=max(squeeze(max(v,[],3)),0); P_lct=P_lct/(max(P_lct(:))+eps); catch, end
try, v=EFsimulation_cnlos_reconstruction(meas_hwT,tofgrid_raw,width,2); P_fk=max(squeeze(max(v,[],3)),0); P_fk=P_fk/(max(P_fk(:))+eps); catch, end
try, v=EFsimulation_cnlos_reconstruction(meas_hwT,tofgrid_raw,width,3); P_phasor=max(squeeze(max(v,[],3)),0); P_phasor=P_phasor/(max(P_phasor(:))+eps); catch, end

%% 11) 显示/网格
P_inva = squeeze(sum(img_inva(td1:td2,:,:),1)); P_inva = P_inva/(max(P_inva(:))+eps);
P_rl1dsolve = squeeze(sum(img_rl1dsolve(td1:td2,:,:),1)); P_rl1dsolve=P_rl1dsolve/(max(P_rl1dsolve(:))+eps);
method_names = {'GRL','Reg-Inv','RL (1D) + Solve','FBP','LCT','f-k','Phasor'};
P_list = {P_grl,P_inva,P_rl1dsolve,P_fbp,P_lct,P_fk,P_phasor};

figure('Name',sprintf('Experiment Multi-Method Compare | %s | window=[%d,%d]', data_file, td1, td2),'Color','w','Position',[60,80,1750,360]);
tiledlayout(1,numel(P_list),'Padding','compact','TileSpacing','compact');
for k=1:numel(P_list)
    nexttile; imagesc(P_list{k}); axis image off; title(method_names{k},'Interpreter','none'); colormap(gca,'turbo');
end

if is_sh && params.sh_enable_grid_search
    [grid_results, grid_labels] = run_sh_param_grid(estimatedImage, estimatedImage2, has_second_pass, td1, td2, td1b, td2b);
    figure('Name',sprintf('SH GRL Param Grid | %s', data_file),'Color','w','Position',[60,470,1800,760]);
    nG=numel(grid_results); ncol=4; nrow=ceil(nG/ncol); tiledlayout(nrow,ncol,'Padding','compact','TileSpacing','compact');
    for i=1:nG, nexttile; imagesc(grid_results{i}); axis image off; colormap(gca,'turbo'); title(grid_labels{i},'Interpreter','none','FontSize',9); end
    assignin('base','grid_results',grid_results); assignin('base','grid_labels',grid_labels);
end

if is_sh && params.sh_enable_secondpass_grid && has_second_pass
    [sp_results, sp_labels] = run_sh_secondpass_grid(data, A_H, At_H, sz, T, params, td1, td2, pk_aux, sh_iters, estimatedImage);
    figure('Name',sprintf('SH Second-pass Grid | %s', data_file),'Color','w','Position',[70,70,1800,760]);
    nS=numel(sp_results); ncol=3; nrow=ceil(nS/ncol); tiledlayout(nrow,ncol,'Padding','compact','TileSpacing','compact');
    for i=1:nS, nexttile; imagesc(sp_results{i}); axis image off; colormap(gca,'turbo'); title(sp_labels{i},'Interpreter','none','FontSize',9); end
    assignin('base','sp_results',sp_results); assignin('base','sp_labels',sp_labels);
end

%% 12) 导出
assignin('base','P_grl',P_grl); assignin('base','P_inva',P_inva); assignin('base','P_rl1dsolve',P_rl1dsolve);
assignin('base','P_fbp',P_fbp); assignin('base','P_lct',P_lct); assignin('base','P_fk',P_fk); assignin('base','P_phasor',P_phasor);
assignin('base','method_names',method_names); assignin('base','P_list',P_list);
exp_meta = struct('data_file',data_file,'root_data',root_data,'window',[td1,td2],'params',params,'width',width,'range',range,'sigma_psf',sigma_psf);
assignin('base','exp_meta',exp_meta);
fprintf('[OK] Workspace exported\n');

%% ===== local =====
function [grid_results, grid_labels] = run_sh_param_grid(estimatedImage, estimatedImage2, has_second_pass, td1, td2, td1b, td2b)
    blends = [0.72,0.85]; topboosts=[0.30,0.40]; sumweights=[0.84,0.90];
    grid_results={}; grid_labels={}; idx=0;
    P_sum_base = squeeze(sum(estimatedImage(td1:td2,:,:),1));
    P_max_base = squeeze(max(estimatedImage(td1:td2,:,:),[],1)); P_max_base=P_max_base/(prctile(P_max_base(:),99.7)+eps);
    for ib=1:numel(blends)
        for it=1:numel(topboosts)
            for isw=1:numel(sumweights)
                idx=idx+1; P_sum=P_sum_base;
                if has_second_pass && ~isempty(estimatedImage2)
                    P_aux = squeeze(sum(estimatedImage2(td1b:td2b,:,:),1)); P_aux=P_aux/(prctile(P_aux(:),99.5)+eps);
                    P_sum = P_sum/(prctile(P_sum(:),99.5)+eps);
                    P_sum = (1-blends(ib))*P_sum + blends(ib)*P_aux;
                    [H2,W2]=size(P_sum); topMask=zeros(H2,1); topMask(1:round(0.55*H2))=1;
                    P_sum = P_sum + topboosts(it)*(topMask*ones(1,W2)).*P_aux;
                end
                P_sum = P_sum/(prctile(P_sum(:),99.7)+eps);
                ws=sumweights(isw); P=ws*P_sum+(1-ws)*P_max_base;
                P=P/(prctile(P(:),99.7)+eps); P=min(max(P,0),1);
                grid_results{idx,1}=P;
                grid_labels{idx,1}=sprintf('blend=%.2f | top=%.2f | sumW=%.2f',blends(ib),topboosts(it),ws);
            end
        end
    end
end

function [sp_results, sp_labels] = run_sh_secondpass_grid(data, A_H, At_H, sz, T, params, td1, td2, pk_aux, sh_iters, est_main)
    pres=[55,65,80]; posts=[85,95,120]; auxW=[0.38,0.50,0.55];
    sp_results={}; sp_labels={}; idx=0;
    for ip=1:numel(pres)
        for io=1:numel(posts)
            td1b=max(params.sh_td1_min,pk_aux-pres(ip)); td2b=min(params.sh_td2_max,pk_aux+posts(io));
            mask2=false(sz); mask2(td1b:td2b,:,:)=true; M2=double(mask2);
            in2=data.*M2; in2(isnan(in2))=0; bg2=median(reshape(in2,T,[]),2); in2=max(in2-reshape(bg2,[T,1,1]),0);
            est2=max(At_H(in2),1e-8); est2=est2/(max(est2(:))+eps)*(max(in2(:))+eps); sens2=max(At_H(M2),1e-8);
            for k=1:round(0.95*sh_iters)
                conv2=A_H(est2)+1e-9; r2=zeros(sz); r2(mask2)=in2(mask2)./conv2(mask2); b2=At_H(r2);
                est2=max(est2.*b2./sens2,0); est2(~isfinite(est2))=0;
                if mod(k,3)==0, est2=smooth3(est2,'gaussian',[1,3,3],0.45); end
            end
            Pm=squeeze(sum(est_main(td1:td2,:,:),1)); Paux=squeeze(sum(est2(td1b:td2b,:,:),1));
            Pm=Pm/(prctile(Pm(:),99.5)+eps); Paux=Paux/(prctile(Paux(:),99.5)+eps);
            for iw=1:numel(auxW)
                idx=idx+1; P=(1-auxW(iw))*Pm + auxW(iw)*Paux;
                P=P/(prctile(P(:),99.7)+eps); P=min(max(P,0),1);
                sp_results{idx,1}=P;
                sp_labels{idx,1}=sprintf('pre=%d post=%d auxW=%.2f',pres(ip),posts(io),auxW(iw));
            end
        end
    end
end

function psf = definePsf(U,V,slope,sigma)
    x=linspace(-1,1,2*U); y=linspace(-1,1,2*U); z=linspace(0,2,2*V);
    [gz,gy,gx]=ndgrid(z,y,x);
    psf=abs(((4*slope)^2).*(gx.^2+gy.^2)-gz);
    psf=double(psf==repmat(min(psf,[],1),[2*V 1 1]));
    psf=reshape(fspecial('gaussian',[size(psf,2),size(psf,3)],sigma),[1,size(psf,2),size(psf,3)]).*psf;
    psf=psf./sum(psf(:,U,U)); psf=psf./norm(psf(:)); psf=circshift(psf,[0 U U]);
end

function [mtx,mtxi] = resamplingOperator(M)
    mtx=sparse([],[],[],M^2,M,M^2); x=1:M^2;
    mtx(sub2ind(size(mtx),x,ceil(sqrt(x))))=1;
    mtx=spdiags(1./sqrt(x)',0,M^2,M^2)*mtx; mtxi=mtx';
    K=log(M)/log(2);
    for k=1:round(K)
        mtx=0.5*(mtx(1:2:end,:)+mtx(2:2:end,:));
        mtxi=0.5*(mtxi(:,1:2:end)+mtxi(:,2:2:end));
    end
end

function data_rl = rl_deconv_timehist_fixedpsf_window(data, h, opts)
    [T,Hn,Wn]=size(data); data=max(data,0);
    if ~isfield(opts,'nIter'), opts.nIter=8; end
    if ~isfield(opts,'eps'), opts.eps=1e-6*max(1,mean(data(:))); end
    if ~isfield(opts,'deconv_range')||isempty(opts.deconv_range), opts.deconv_range=[1,T]; end
    if ~isfield(opts,'keep_full_length'), opts.keep_full_length=true; end
    td1=max(1,round(opts.deconv_range(1))); td2=min(T,round(opts.deconv_range(2))); Lw=td2-td1+1;
    h=max(h(:),0);
    if numel(h)<Lw, hh=zeros(Lw,1); hh(1:numel(h))=h; h=hh; elseif numel(h)>Lw, h=h(1:Lw); end
    h=h/max(sum(h),eps);
    data_rl=data;
    for iy=1:Hn
        for ix=1:Wn
            y=data(:,iy,ix); ywin=y(td1:td2); x=max(ywin,0);
            for it=1:opts.nIter
                yhat=conv(x,h,'same'); ratio=ywin./(yhat+opts.eps); corrT=conv(ratio,flipud(h),'same'); x=max(x.*corrT,0);
            end
            if opts.keep_full_length, yout=y; yout(td1:td2)=x; else, yout=zeros(T,1); yout(td1:td2)=x; end
            data_rl(:,iy,ix)=yout;
        end
    end
end
