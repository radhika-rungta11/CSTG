#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE ####################################
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################




import torch
import math
import time 
import torch.nn.functional as F
import time 




from scene.oursfull import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getProjectionMatrixCV, focal2fov, fov2focal

def train_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None, rvq_iter=False, compute_alpha=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    shs = None
    tforpoly = trbfdistanceoffset.detach()
    if rvq_iter:
        vq_rot, _, _ = pc.vq_rot(pc._rotation.unsqueeze(0))
        vq_sca, _, _ = pc.vq_sca(pc._scaling.unsqueeze(0))

        vq_tfea, _, _ = pc.vq_tfea(pc._features_t.unsqueeze(0))
        vq_ome, _, _ = pc.vq_ome(pc._omega.unsqueeze(0))
        
        xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        colors_precomp = pc.mlp_head(pc.recolor(xyz)).float()
        colors_precomp = pc.get_features(colors_precomp, vq_tfea[0], tforpoly)
    
        means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    
        pc.delta_t = tforpoly
        rotations = pc.get_rotation(tforpoly, vq_rot[0], vq_ome[0])
        scales = pc.scaling_activation(vq_sca[0])

    else:
        xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        colors_precomp = pc.mlp_head(pc.recolor(xyz)).float()#.unsqueeze(1)
        colors_precomp = pc.get_features(colors_precomp, pc._features_t, tforpoly)
    
        means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
        rotations = pc.get_rotation(tforpoly, pc._rotation, pc._omega)
        scales = pc.get_scaling
    
    mask = ((torch.sigmoid(pc._mask) > 0.01).float()- torch.sigmoid(pc._mask)).detach() + torch.sigmoid(pc._mask)
    opacity = opacity*mask
    scales = scales*mask
    
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp.float(),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)

    rendered_alpha = None
    if compute_alpha:
        # Second rasterizer pass for accumulated alpha. Set every Gaussian's
        # "color" to ones and bg to zero. The rasterizer's output channel value
        # then equals  sum_i(opacity_i * T_{<i}) = 1 - T_final = alpha
        # (alpha-blending identity, color-blind).
        # Skip pc.rgbdecoder — it'd map 9-ch features → 3-ch RGB and corrupt alpha.
        NUM_CH = colors_precomp.shape[1]
        ones_colors = torch.ones((means3D.shape[0], NUM_CH),
                                 device=means3D.device, dtype=colors_precomp.dtype)
        zero_bg = torch.zeros(NUM_CH, device=means3D.device, dtype=bg_color.dtype)
        raster_settings_alpha = GRsetting(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy,
            bg=zero_bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
        )
        rasterizer_alpha = GRzer(raster_settings=raster_settings_alpha)
        alpha_raw, _, _ = rasterizer_alpha(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=ones_colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        # All channels equal alpha; take channel 0 → (1, H, W)
        rendered_alpha = alpha_raw[0:1].clamp(0.0, 1.0)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "alpha": rendered_alpha,}


def test_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    startime = time.time()

    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc._opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    
    colors_precomp = pc.get_features(pc._features_dc, pc._features_t, tforpoly)

    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly, pc._rotation, pc._omega) # to try use 
    

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp.float(),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    torch.cuda.synchronize()
    duration = time.time() - startime 

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration":duration}


def train_stg_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None, rvq_iter=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}

def test_stg_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    startime = time.time()

    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    torch.cuda.synchronize()
    duration = time.time() - startime 

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration":duration}
