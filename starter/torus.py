"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio

from starter.utils import  get_points_renderer


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_torus(image_size=256, num_samples=200, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    major_radius = 1.0  
    minor_radius = 0.3  

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (major_radius + minor_radius * torch.cos(Theta)) * torch.cos(Phi)
    y = (major_radius + minor_radius * torch.cos(Theta)) * torch.sin(Phi)
    z = minor_radius * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)

    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(torus_point_cloud, cameras=cameras)

    num_frames = 60
    fps = 15
    output_path = "outputs/torus_video.gif"
    
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
        rend = renderer(torus_point_cloud, cameras=cameras, lights=lights)
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
    parser.add_argument("--output_path", type=str, default="outputs/torus.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":    
        image = render_torus(image_size=args.image_size, num_samples=args.num_samples)
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)

