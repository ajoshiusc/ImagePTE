clc;clear all;close all;restoredefaultpath;
%addpath(genpath('/big_disk/ajoshi/coding_ground/bfp/supp_data'))
addpath(genpath('/home/ajoshi/projects/bfp/src'));

%studydir='/ImagePTE1/ajoshi/ADHD_Peking_bfp';
studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1';
sessionid='rest';

a=dir([studydir,'/TB*']);

config.FSLPATH = '/home/ajoshi/webware/fsl';
config.FSLOUTPUTTYPE='NIFTI_GZ';
config.AFNIPATH = '/home/ajoshi/abin';
config.FSLRigidReg=1;
config.MultiThreading=0;
config.BFPPATH='/home/ajoshi/projects/bfp';

%parpool(6);
for j=1:length(a)
    subid = a(j).name;
    fmribase = fullfile(studydir,subid,'BFP',subid,'func',[subid,'_rest_bold']);
    anatbase = fullfile(studydir,subid,'BFP',subid,'anat',[subid,'_T1w']);
    get_alff_gord(config, fmribase, anatbase);
    gen_brainordinates_alff('/home/ajoshi/BrainSuite21a', anatbase, fmribase, 'ALFF_Z');
    gen_brainordinates_alff('/home/ajoshi/BrainSuite21a', anatbase, fmribase, 'ALFF');
    gen_brainordinates_alff('/home/ajoshi/BrainSuite21a', anatbase, fmribase, 'fALFF');
    j
end


