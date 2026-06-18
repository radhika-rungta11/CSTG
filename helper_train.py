#
# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =============================================

# This license is additionally subject to the following restrictions:

# Licensor grants non-exclusive rights to use the Software for research purposes
# to research users (both academic and industrial), free of charge, without right
# to sublicense. The Software may be used "non-commercially", i.e., for research
# and/or evaluation purposes only.

# Subject to the terms and conditions of this License, you are granted a
# non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
# publicly display, publicly perform and distribute its Work and any resulting
# derivative works in any form.
#

import torch
import numpy as np
import torch
from simple_knn._C import distCUDA2
import os 
import json 
import cv2
from script.pre_immersive_distorted import SCALEDICT 


def getrenderpip(option="train_ours_full"):
    print("render option", option)
    from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings 
    from diff_gaussian_rasterization_ch9 import GaussianRasterizer  
    if option == "train_stg_full":
        from thirdparty.gaussian_splatting.renderer import train_stg_full 
        return train_stg_full, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_ours_full":
        from thirdparty.gaussian_splatting.renderer import train_ours_full
        return train_ours_full, GaussianRasterizationSettings, GaussianRasterizer
    
    elif option == "test_stg_full":
        from thirdparty.gaussian_splatting.renderer import test_stg_full
        return test_stg_full, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_ours_full": # forward only 
        from thirdparty.gaussian_splatting.renderer import test_ours_full
        return test_ours_full, GaussianRasterizationSettings, GaussianRasterizer
    else:
        raise NotImplementedError("Rennder {} not implemented".format(option))
    
def getmodel(model="oursfull"):
    if model == "stg":
        from  thirdparty.gaussian_splatting.scene.stgfull import GaussianModel
    elif model == "ours":
        from  thirdparty.gaussian_splatting.scene.oursfull import GaussianModel
    else:
    
        raise NotImplementedError("model {} not implemented".format(model))
    return GaussianModel

def getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii):
    if opt.reg == 1: # add optical flow loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._motion) / gaussians._motion.shape[0]
    elif opt.reg == 0 :
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))   
    elif opt.reg == 9 : #regulizor on the rotation
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._omega[radii>0]**2)
    elif opt.reg == 10 : #regulizor on the rotation
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._motion[radii>0]**2)
    elif opt.reg == 4:
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians.get_scaling) / gaussians._motion.shape[0]
    elif opt.reg == 5:
        loss = Ll1  
    elif opt.reg == 6 :
        ratio = torch.mean(gt_image) - 0.5 + opt.lambda_dssim
        ratio = torch.clamp(ratio, 0.0, 1.0)
        loss = (1.0 - ratio) * Ll1 + ratio * (1.0 - ssim(image, gt_image))
    elif opt.reg == 7 :
        Ll1 = Ll1 / (torch.mean(gt_image) * 2.0) # normalize L1 loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    elif opt.reg == 8 :
        N = gaussians._xyz.shape[0]
        mean = torch.mean(gaussians._xyz, dim=0, keepdim=True)
        varaince = (mean - gaussians._xyz)**2 #/ N
        loss = (1.0 - opt.lambda_dssim) * Ll1  + 0.0002*opt.lambda_dssim * torch.sum(varaince) / N
    return loss 


def freezweights(model, screenlist):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad = torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  

def freezweightsbymask(model, screenlist, mask):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad =  mask.unsqueeze(1)*grad_tensor #torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  


def freezweightsbymasknounsqueeze(model, screenlist, mask):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        if grad_tensor is None:
            continue
        newgrad =  mask*grad_tensor #torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return


def removeminmax(gaussians, maxbounds, minbounds):
    maxx, maxy, maxz = maxbounds
    minx, miny, minz = minbounds
    xyz = gaussians._xyz
    mask0 = xyz[:,0] > maxx.item()
    mask1 = xyz[:,1] > maxy.item()
    mask2 = xyz[:,2] > maxz.item()

    mask3 = xyz[:,0] < minx.item()
    mask4 = xyz[:,1] < miny.item()
    mask5 = xyz[:,2] < minz.item()
    mask =  logicalorlist([mask0, mask1, mask2, mask3, mask4, mask5])
    gaussians.prune_points(mask) 
    torch.cuda.empty_cache()


def _mcmc_refine_step(opt, gaussians, iteration, scene):
    """Run one MCMC refine step (relocate + sample_add) if it's a scheduled iter.

    Replaces the heuristic densify_pruneclone + reset_opacity path. Returns the
    pair (n_relocated, n_added) — both 0 if the step is skipped.
    """
    if (iteration < opt.mcmc_refine_start
            or iteration > opt.mcmc_refine_stop
            or iteration >= opt.iterations
            or iteration % opt.mcmc_refine_every != 0):
        return 0, 0
    n_rel = gaussians._mcmc_relocate(opt.mcmc_min_opacity)
    n_new = gaussians._mcmc_add(opt.mcmc_cap_max)
    if n_rel or n_new:
        scene.recordpoints(
            iteration,
            f"mcmc relocated={n_rel} added={n_new} total={gaussians.get_xyz.shape[0]}",
        )
        if os.environ.get("MCMC_SANITY") == "1":
            _mcmc_sanity(gaussians, iteration)
    return n_rel, n_new


def _mcmc_sanity(g, it):
    """Opt-in diagnostic (MCMC_SANITY=1). Verifies per-Gaussian tensor sizes,
    optimizer-state alignment, and value finiteness after each refine. Adds a
    full-tensor isfinite pass at 3M+ Gaussians, so it's off in production."""
    n = g._xyz.shape[0]
    bad = []
    for attr, _ in g._MCMC_PARAMS:
        p = getattr(g, attr)
        if p.shape[0] != n:
            bad.append(f"{attr} shape[0]={p.shape[0]} != {n}")
        elif not torch.isfinite(p).all():
            bad.append(f"{attr} has NaN/Inf")
    for pg in g.optimizer.param_groups:
        if pg["name"] not in {nm for _, nm in g._MCMC_PARAMS}:
            continue
        p = pg["params"][0]
        attr = next(a for a, nm in g._MCMC_PARAMS if nm == pg["name"])
        if p is not getattr(g, attr):
            bad.append(f"opt group {pg['name']} param IS NOT model.{attr}")
            continue
        st = g.optimizer.state.get(p, None)
        if st is not None:
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st and st[k].shape != p.shape:
                    bad.append(f"opt {pg['name']}.{k} shape={st[k].shape} != {p.shape}")
    if bad:
        print(f"[mcmc-sanity] iter={it}: " + "; ".join(bad), flush=True)


def _densify_refine_step(opt, gaussians, iteration, scene):
    """Dispatch to the configured densifier: MCMC (default) or taming.

    The taming densifier lives in the self-contained `taming/` package; it is
    imported lazily so the MCMC path carries no extra import cost."""
    if getattr(opt, "densify_mode", "mcmc") == "taming":
        from taming import densify_step as taming_densify_step
        taming_densify_step(opt, gaussians, iteration, scene)
    else:
        _mcmc_refine_step(opt, gaussians, iteration, scene)


def controlgaussians(opt, gaussians, densify, iteration, scene,  visibility_filter, radii, viewspace_point_tensor, flag, traincamerawithdistance=None, maxbounds=None, minbounds=None):
    if densify == 1: # n3d
        if iteration < opt.mcmc_refine_stop:
            if iteration ==  8001 : # 8001
                omegamask = gaussians.zero_omegabymotion() # 1 we keep omega, 0 we freeze omega
                gaussians.omegamask  = omegamask
                scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            elif iteration > 8001: # 8001
                if gaussians.omegamask is None or gaussians.omegamask.shape[0] != gaussians.get_xyz.shape[0]:
                    omegamask = gaussians.zero_omegabymotion()
                    gaussians.omegamask = omegamask
                    scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
                freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
                rotationmask = torch.logical_not(gaussians.omegamask)
                freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)
            _densify_refine_step(opt, gaussians, iteration, scene)
        else:
            try:
                freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
                rotationmask = torch.logical_not(gaussians.omegamask)
                freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)
            except:
                pass
            if iteration % 1000 == 500 :
                zmask = gaussians._xyz[:,2] < 4.5  # n3d-specific z-prune for stability
                gaussians.prune_points(zmask)
                torch.cuda.empty_cache()
            if iteration == 10000:
                removeminmax(gaussians, maxbounds, minbounds)
        return flag

    elif densify == 2: # n3d
        if iteration < opt.mcmc_refine_stop:
            if iteration ==  8001 : # 8001
                omegamask = gaussians.zero_omegabymotion()
                gaussians.omegamask  = omegamask
                scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            elif iteration > 8001: # 8001
                if gaussians.omegamask is None or gaussians.omegamask.shape[0] != gaussians.get_xyz.shape[0]:
                    omegamask = gaussians.zero_omegabymotion()
                    gaussians.omegamask = omegamask
                    scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
                freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
                rotationmask = torch.logical_not(gaussians.omegamask)
                freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)
            _densify_refine_step(opt, gaussians, iteration, scene)
        else:
            if iteration % 1000 == 500 :
                zmask = gaussians._xyz[:,2] < 4.5  # for stability
                gaussians.prune_points(zmask)
                torch.cuda.empty_cache()
        return flag


    elif densify == 3: # techni
        if iteration < opt.mcmc_refine_stop and iteration < opt.iterations:
            _densify_refine_step(opt, gaussians, iteration, scene)
        else:
            if gaussians.omegamask is None or gaussians.omegamask.shape[0] != gaussians.get_xyz.shape[0]:
                omegamask = gaussians.zero_omegabymotion()
                gaussians.omegamask = omegamask
                scene.recordpoints(iteration, "seperate omega" + str(torch.sum(omegamask).item()))
            freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
            rotationmask = torch.logical_not(gaussians.omegamask)
            freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)
            if iteration == 10000:
                removeminmax(gaussians, maxbounds, minbounds)
        return flag
    


def logicalorlist(listoftensor):
    mask = None 
    for idx, ele in enumerate(listoftensor):
        if idx == 0 :
            mask = ele 
        else:
            mask = torch.logical_or(mask, ele)
    return mask 



def recordpointshelper(model_path, numpoints, iteration, string):
    txtpath = os.path.join(model_path, "exp_log.txt")
    
    with open(txtpath, 'a') as file:
        file.write("iteration at "+ str(iteration) + "\n")
        file.write(string + " pointsnumber " + str(numpoints) + "\n")




def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0




def reloadhelper(gaussians, opt, maxx, maxy, maxz,  minx, miny, minz):
    givenpath = opt.prevpath
    if opt.loadall == 0:
        gaussians.load_plyandminmax(givenpath, maxx, maxy, maxz,  minx, miny, minz)
    elif opt.loadall == 1 :
        gaussians.load_plyandminmaxall(givenpath, maxx, maxy, maxz,  minx, miny, minz)
    elif opt.loadall == 2 :        
        gaussians.load_ply(givenpath)
    elif opt.loadall == 3:
        gaussians.load_plyandminmaxY(givenpath, maxx, maxy, maxz,  minx, miny, minz)

    gaussians.max_radii2D =  torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    return 

def getfisheyemapper(folder, cameraname):
    parentfolder = os.path.dirname(folder)
    distoritonflowpath = os.path.join(parentfolder, cameraname + ".npy")
    distoritonflow = np.load(distoritonflowpath)
    distoritonflow = torch.from_numpy(distoritonflow).unsqueeze(0).float().cuda()
    return distoritonflow











def undistortimage(imagename, datasetpath,data):
    


    video = os.path.dirname(datasetpath) # upper folder 
    with open(os.path.join(video + "/models.json"), "r") as f:
                meta = json.load(f)

    for idx , camera in enumerate(meta):
        folder = camera['name'] # camera_0001
        view = camera
        intrinsics = np.array([[view['focal_length'], 0.0, view['principal_point'][0]],
                            [0.0, view['focal_length'], view['principal_point'][1]],
                            [0.0, 0.0, 1.0]])
        dis_cef = np.zeros((4))

        dis_cef[:2] = np.array(view['radial_distortion'])[:2]
        if folder != imagename:
             continue
        print("done one camera")
        map1, map2 = None, None
        sequencename = os.path.basename(video)
        focalscale = SCALEDICT[sequencename]
 
        h, w = data.shape[:2]


        image_size = (w, h)
        knew = np.zeros((3, 3), dtype=np.float32)

def trbfunction(x):
    return torch.exp(-1*x.pow(2))


def save_checkpoint(gaussians, opt, flag, iteration, flagems, emscnt, lasterems, model_path):
    """Save full training state for resume."""
    import torch.nn as nn
    ckpt_dir = os.path.join(model_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{iteration:06d}.pth")

    param_names = ['_xyz', '_features_dc', '_features_t', '_opacity', '_scaling',
                   '_rotation', '_omega', '_trbf_center', '_trbf_scale', '_motion', '_mask']
    gaussian_params = {n: getattr(gaussians, n).detach().cpu()
                       for n in param_names if hasattr(gaussians, n) and getattr(gaussians, n) is not None}

    net_state = {name: getattr(gaussians, name).state_dict()
                 for name in ['recolor', 'mlp_head', 'rgbdecoder']
                 if hasattr(gaussians, name)}

    torch.save({
        'iteration': iteration,
        'flag': flag,
        'flagems': flagems,
        'emscnt': emscnt,
        'lasterems': lasterems,
        'omegamask': gaussians.omegamask.cpu() if gaussians.omegamask is not None else None,
        'spatial_lr_scale': gaussians.spatial_lr_scale,
        'gaussian_params': gaussian_params,
        'densify_state': {
            'max_radii2D': gaussians.max_radii2D.detach().cpu(),
            'xyz_gradient_accum': gaussians.xyz_gradient_accum.detach().cpu(),
            'denom': gaussians.denom.detach().cpu(),
        },
        'net_state': net_state,
        'optimizer': gaussians.optimizer.state_dict(),
        'optimizer_net': gaussians.optimizer_net.state_dict(),
        'scheduler_net': gaussians.scheduler_net.state_dict(),
    }, ckpt_path)
    print(f"\n[ITER {iteration}] Checkpoint saved: {ckpt_path}")
    return ckpt_path


def load_checkpoint(ckpt_path, gaussians, opt):
    """Restore full training state from checkpoint."""
    import torch.nn as nn
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"Resuming from checkpoint: {ckpt_path} (iteration {ckpt['iteration']})")

    # Restore gaussian parameters
    for name, tensor in ckpt['gaussian_params'].items():
        setattr(gaussians, name, nn.Parameter(tensor.cuda()))

    gaussians.spatial_lr_scale = ckpt['spatial_lr_scale']

    # Restore neural networks before training_setup
    for name, state in ckpt['net_state'].items():
        if hasattr(gaussians, name):
            getattr(gaussians, name).load_state_dict(state)

    # Re-create optimizers (they reference the restored parameter objects)
    gaussians.training_setup(opt)

    # Restore optimizer states
    gaussians.optimizer.load_state_dict(ckpt['optimizer'])
    gaussians.optimizer_net.load_state_dict(ckpt['optimizer_net'])
    gaussians.scheduler_net.load_state_dict(ckpt['scheduler_net'])

    # Restore densification accumulators
    gaussians.max_radii2D = ckpt['densify_state']['max_radii2D'].cuda()
    gaussians.xyz_gradient_accum = ckpt['densify_state']['xyz_gradient_accum'].cuda()
    gaussians.denom = ckpt['densify_state']['denom'].cuda()

    if ckpt['omegamask'] is not None:
        gaussians.omegamask = ckpt['omegamask'].cuda()

    return (ckpt['iteration'], ckpt['flag'],
            ckpt['flagems'], ckpt['emscnt'], ckpt['lasterems'])


def find_latest_checkpoint(model_path):
    """Return path to the latest checkpoint in model_path/checkpoints/, or None."""
    ckpt_dir = os.path.join(model_path, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".pth"))
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else None
