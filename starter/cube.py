import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio

from starter.utils import get_device, get_mesh_renderer

device = None

if device is None:
        device = get_device()


def render_tetrahedron(output_path, image_size=256):
    
    renderer = get_mesh_renderer(image_size=image_size)


    vertices = torch.tensor([[0.0, 0.0, 0.0],  [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  [0.0, 0.0, 1.0],  [1.0, 0.0, 1.0],  [1.0, 1.0, 1.0],  [0.0, 1.0, 1.0] ])

    vertices = vertices.unsqueeze(0)

    faces = torch.tensor([[0, 1, 2],  [0, 2, 3],  [4, 5, 6],  [4, 6, 7],  [0, 3, 7],  [0, 7, 4],  [1, 2, 6],  [1, 6, 5],  [2, 3, 7],  [2, 7, 6],  [0, 1, 5],  [0, 5, 4] ])

    faces = faces.unsqueeze(0)

    color = [1.0, 0.0, 0.0]  
    textures = torch.ones(vertices.shape) * torch.tensor(color)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures)
    )
    mesh = mesh.to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    num_frames = 60
    fps = 15

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs/cube_rotation.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()


    render_tetrahedron(args.output_path, args.image_size)
