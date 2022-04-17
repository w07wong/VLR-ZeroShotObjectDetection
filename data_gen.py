"""
This script generates different views for a YCB 3D model
"""

import os
import argparse

import torch
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (RasterizationSettings, 
        MeshRenderer, 
        MeshRasterizer, 
        HardPhongShader,
        Materials,
        TexturesUV,
        TexturesVertex,
        BlendParams)

from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.transforms import euler_angles_to_matrix
from torchvision.utils import save_image

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights, blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))),
    )
    return renderer


def generate_model_views(model_dir, output_dir, image_size=(480, 640)):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    renderer = get_mesh_renderer(image_size=image_size)
   
    mesh_path = model_dir + '/' + 'textured_simple.obj'
    mesh = load_objs_as_meshes([mesh_path], device=device).to(device)


    # Mean-center the mesh and scale
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    mesh.offset_verts_(-center)
    scale = max((verts - center).abs().max(0)[0])
    mesh.scale_verts_((1.0 / float(scale)));

    # Rotate mesh
    verts = mesh.verts_padded()[0]
    rot_deg = torch.tensor([-90*torch.pi/180, 0, 0]).to(device)
    rot_mtx = euler_angles_to_matrix(rot_deg, convention="XYZ")
    verts = (rot_mtx @ verts.T).T
    mesh = mesh.update_padded(verts.unsqueeze(0))
     
    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Turn-table rendering
    k = 0

    # Elevation
    for i, elv in enumerate([0, 30, 45]):

        # Azimuth
        for deg in range(0, 361, 10):

            R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=4.0, elev=elv, azim=deg, degrees=True, device=device)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
            rend = renderer(mesh, cameras=cameras, lights=lights).cpu()[0, ..., :3]

            # [H, W, 3] --> [3, H, W]
            rend = rend.permute(2,0,1)
            save_image(rend, output_dir + f'/{str(k).zfill(4)}.png')
            k += 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate views for a YCB model.')
    parser.add_argument('--model-dir', type=str, default='./YCB/models/002_master_chef_can/', help='Path to directory of a model')
    parser.add_argument('--output-dir', type=str, default='./YCB/models/target/', help='Path to output directory for the model renders')
    parser.add_argument('--image-size', type=int, nargs='+', help='Image size')

    args = parser.parse_args()

    generate_model_views(args.model_dir, args.output_dir, tuple(args.image_size))




