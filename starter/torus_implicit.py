"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""

import pickle
import mcubes
import pytorch3d
import torch
import imageio
import matplotlib.pyplot as plt
import argparse

from starter.utils import get_device, get_mesh_renderer

def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_mesh_torus_implicit(image_size=512, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    
    min_value = -3
    max_value = 3

    major_radius = 1.0  
    minor_radius = 0.3  
    
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    voxels = (torch.sqrt(X**2 + Y**2) - major_radius)**2 + Z**2 - minor_radius**2
    
    vertices, faces = mcubes.marching_cubes(voxels.cpu().numpy(), isovalue=0)
    
    vertices = torch.tensor(vertices).float().to(device)
    faces = torch.tensor(faces.astype(int)).to(device)
    
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    
    rend = renderer(mesh, cameras=cameras, lights=lights)

    num_frames = 60
    fps = 15
    output_path = "outputs/torusImplict_video.gif"

    elevations = torch.linspace(0, 360, num_frames, device=device)
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    for elevation, azimuth in zip(elevations, azimuths):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3.0,
            elev=elevation,
            azim=azimuth
        )
        cameras.R = R.to(device)
        cameras.T = T.to(device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy() 
        images.append(rend)

    imageio.mimsave(output_path, images, fps=fps) 
    
    return rend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="output/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "implicit":
        image = render_mesh_torus_implicit(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)

