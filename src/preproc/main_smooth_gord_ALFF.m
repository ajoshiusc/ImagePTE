clc;clear all;close all;
addpath(genpath('/ImagePTE1/ajoshi/code_farm/bfp/src'));

inp_dir='/ImagePTE1/ajoshi/maryland_rao_v1_bfp/maryland_v1_alff';
out_dir='/ImagePTE1/ajoshi/maryland_rao_v1_bfp/maryland_v1_alff_smooth';

d = dir(inp_dir);

left_surf = '/home/ajoshi/projects/bfp/supp_data/bci32kleft.dfs';
lsurf = readdfs(left_surf);
right_surf = '/home/ajoshi/projects/bfp/supp_data/bci32kright.dfs';
rsurf = readdfs(right_surf);
NV=length(lsurf.vertices);
parpool(12);
parfor i=3:length(d)

    fname = fullfile(inp_dir,d(i).name);
    outfname = fullfile(out_dir,[d(i).name(1:end-3),'smooth.mat']);
    if ~exist(outfname,'file')
        process_data(fname,outfname,NV,lsurf,rsurf);
    end

end


function process_data(fname,outfname,NV,lsurf,rsurf)
    load(fname);

    left_data = data(1:NV);
    right_data = data(NV+1:2*NV);

    data(1:NV)=smooth_surf_function(lsurf,left_data);
    data(NV+1:2*NV)=smooth_surf_function(rsurf,right_data);

    save(outfname,'data');
end