%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show the most basic usage of MCXLAB.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfgs
cfg.nphoton=1e6;
cfg.vol=ones(250,250,150);
% cfg.shapes=['{"Shapes":[{"Grid": {"Size": [60, 60, 60], "Tag": 0}},' ...
%     '{"Cylinder": {"Tag":2, "C0": [0,100.5,100.5], "C1": [200,100.5,100.5], "R": 20}}]}'];
cfg.shapes=['{"Shapes":[' ...
    '{"Grid": {"Size": [250, 250, 150], "Tag": 0}},' ...
    '{"Subgrid": {"Size": [250, 250, 4], "O": [0, 0, 0], "Tag": 1}},' ...
    '{"Subgrid": {"Size": [250, 250, 4], "O": [0, 0, 4], "Tag": 2}},' ...
    '{"Subgrid": {"Size": [250, 250, 142], "O": [0, 0, 8], "Tag": 3}},' ...
    '{"Cylinder": {"Tag":4, "C0": [0,125,42], "C1": [250,125,42], "R": 18}},' ...
    '{"Cylinder": {"Tag":5, "C0": [0,107,64], "C1": [250,107,64], "R": 10}}' ...
    ']}'];
cfg.unitinmm=0.25;
cfg.srcpos=[85 125 0];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.issrcfrom0=1;
cfg.prop=[0 0 1 1; 0.25 12 0.8 1.4; 0.1 17.5 0.9 1.4; 0.05 6.08 0.9 1.4; 0.4 5 0.9 1.4; 0.3 5 0.9 1.4];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
% calculate the flux distribution with the given config
cfg.detpos=[165 125 0 1.47];
%cfg.savedetflag='dsp';
[flux, detp, vol, seeds]=mcxlab(cfg);

newcfg=cfg;
newcfg.seed=seeds.data;
newcfg.outputtype='jacobian';
newcfg.detphotons=detp.data;
[flux2, detp2, vol2, seeds2]=mcxlab(newcfg);
jac=sum(flux2.data,4);
imagesc(log10(abs(squeeze(jac(:,125,:)))));
set(gca,'dataAspectRatio',[1 1 1]);