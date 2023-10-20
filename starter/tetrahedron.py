import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import numpy as np

from starter.utils import get_device, get_mesh_renderer

device = None

if device is None:
        device = get_device()

def create_tetrahedron_mesh():
    vertices = torch.tensor([[0.0, 0.0, 0.0],  [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],  [0.5, 0.5, 1.0] ])

    vertices = vertices.unsqueeze(0)

    faces = torch.tensor([[0, 1, 2],  [0, 1, 3],  [0, 2, 3],  [1, 2, 3] ])

    faces = faces.unsqueeze(0)

    color = [1.0, 0.0, 0.0]  
    textures = torch.ones(vertices.shape) * torch.tensor(color)

    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=pytorch3d.renderer.TexturesVertex(textures))
    mesh = mesh.to(device)

    return mesh

def render_tetrahedron(mesh, output_path, image_size=256):
    device = get_device()
    renderer = get_mesh_renderer(image_size=image_size)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    num_frames = 60
    fps = 15
    duration=3,

    elevations = torch.linspace(0, 360, num_frames, device=device)
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []

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

    # renders = []
    # for fov in tqdm(fovs):
    #     distance = 3  # TODO: change this.
    #     T = [[0, 0, 3]]  # TODO: Change this.
    #     cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
    #     rend = renderer(mesh, cameras=cameras, lights=lights)
    #     rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
    #     renders.append(rend)

    # images = []
    # for i, r in enumerate(renders):
    #     image = Image.fromarray((r * 255).astype(np.uint8))
    #     draw = ImageDraw.Draw(image)
    #     draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
    #     images.append(np.array(image))
    # imageio.mimsave(output_path, images, fps=(num_frames / duration))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs/tetrahedron_rotation.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    tetrahedron_mesh = create_tetrahedron_mesh()
    render_tetrahedron(tetrahedron_mesh, args.output_path, args.image_size)
