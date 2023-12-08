%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show the most basic usage of MCXLAB.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfgs
cfg.nphoton=1e6;
cfg.vol=zeros(200,200,200);
cfg.shapes=['{"Shapes":[{"ZLayers":[[1,20,1],[21,32,4],[33,200,3]]},' ...
    '{"Cylinder": {"Tag":2, "C0": [0,100.5,100.5], "C1": [200,100.5,100.5], "R": 20}}]}'];
cfg.unitinmm=0.005;
cfg.srcpos=[60 100 0];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.issrcfrom0=1;
cfg.prop=[0.0000         0.0    1.0000    1
    0.5    20    0.9    1.3700
   23.0543    9.3985    0.9000    1.3700
    0.0458   35.6541    0.9000    1.3700
    1.6572   37.5940    0.9000    1.3700];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.debuglevel = 'p';
cfg.seed = -1;
% calculate the flux distribution with the given config
cfg.detpos=[140 100 0 1.47];
%cfg.savedetflag='dsp';
[flux, detp, vol, seeds]=mcxlab(cfg);

% newcfg=cfg;
% newcfg.seed=seeds.data;
% newcfg.outputtype='jacobian';
% newcfg.detphotons=detp.data;
% [flux2, detp2, vol2, seeds2]=mcxlab(newcfg);
% jac=sum(flux2.data,4);
% imagesc(log10(abs(squeeze(jac(:,100,:)))));
% set(gca,'dataAspectRatio',[1 1 1]);