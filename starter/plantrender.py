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
from PIL import Image, ImageDraw

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image

def load_rgbd_data(path="data/rgbd_data.pkl", image_size=512, duration=200,device = None, output_path="outputs/plant1.gif", ):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if device is None:
        device = get_device()

    points_image1, rgba_image1 = unproject_depth_image(torch.tensor(data['rgb1']), torch.tensor(data['mask1']), torch.tensor(data['depth1']), data['cameras1'])
    point_cloud_image1 = pytorch3d.structures.Pointclouds(points=[points_image1], features=[rgba_image1]).to(device)
    
    points_image2, rgba_image2 = unproject_depth_image(torch.tensor(data['rgb2']), torch.tensor(data['mask2']), torch.tensor(data['depth2']), data['cameras2'])
    point_cloud_image2 = pytorch3d.structures.Pointclouds(points=[points_image2], features=[rgba_image2]).to(device)

    points_combined = torch.cat((points_image1, points_image2), dim=0)
    rgba_combined = torch.cat((rgba_image1, rgba_image2), dim=0)
    point_cloud_combined3 = pytorch3d.structures.Pointclouds(points=[points_combined], features=[rgba_combined]).to(device)

    num_frames = 60
    fps = 15

    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)

    elevations = torch.linspace(0, 360, num_frames, device=device)
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=[[0, 0, 6]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    for elevation, azimuth in zip(elevations, azimuths):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3.0,
            elev=elevation,
            azim=azimuth
        )
        cameras.R = R.to(device)
        cameras.T = T.to(device)
        rend = renderer(point_cloud_image1, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy() 
        images.append(rend)

    imageio.mimsave(output_path, images, fps=fps)

    return rend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="rgbd",
        choices=["rgbd","point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="outputs/plant1.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "rgbd":
        image = load_rgbd_data()
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)
