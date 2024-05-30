import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturesVertex, TexturesAtlas
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.renderer import HardFlatShader
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.io import load_objs_as_meshes

# local imports
from models.gcn import GCN
from models.discriminator import Discriminator
from utils.mesh_utils import sparse_mx_to_torch_sparse_tensor
from utils.mesh_utils import mesh_features, mesh_features_dual

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# general parameters
save_path = 'checkpoints'

# 3d object
obj_path = 'dataset/test/chair.obj'
verts, faces, aux = load_obj(obj_path, device=device,create_texture_atlas=True)
mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])

# extract feature tensor
# features_tensor, adj_tensor = mesh_features('dataset','chair.obj')
features_tensor, adj_tensor = mesh_features_dual('dataset', mesh)
features_tensor = features_tensor.to(device)
adj_tensor = adj_tensor.to(device)

# network parameters
in_features = features_tensor.shape[1]
n_hidden = 64
out_features = 3
noise_dim = 8
dropout = 0.5
num_epochs = 1
criterion = torch.nn.BCELoss()

# Discriminaotr
netD = Discriminator()

# Generator: gcn model
netG = GCN(in_features=in_features, n_hidden=n_hidden, out_features=out_features,
               noise_dim=noise_dim, dropout=dropout).to(device)

# print(torch.cuda.memory_summary(device=device, abbreviated=False))

print("feature tensor: {}".format(features_tensor.shape))

noise = torch.randn(noise_dim).to(device)
texture = netG(features_tensor, adj_tensor, noise).to(device).unsqueeze(dim=0)

batch_size = 1
image_size = 800
out_res = 1


distance = 1.0
elevation = torch.FloatTensor([41.5770]) #torch.FloatTensor(batch_size).uniform_(0, 90)
azimuth = torch.FloatTensor([33.2110]) #torch.FloatTensor(batch_size).uniform_(0, 90)

print("elevation: {}".format(elevation))
print("azimuth: {}".format(azimuth))

R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)


cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

adjusted_bin_size = 0  # Set to 0 for naive rasterization
adjusted_max_faces_per_bin = 1000  # Increase this based on your mesh complexity
raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                    cull_backfaces=True, bin_size=adjusted_bin_size, max_faces_per_bin=adjusted_max_faces_per_bin)

# Place a point light in front of the object: for now let's put at the same location of camera:
# so the rendered image will get some
# light, we can also place them at s*camera_location where s is some scalar
lights = PointLights(location=[[1.0, 1.0, 1.0]], ambient_color=((0.5, 0.5, 0.5),),
                    diffuse_color=((0.4, 0.4, 0.4),),
                    specular_color=((0.1, 0.1, 0.1),), device=str(device))

# Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
# interpolate the texture uv coordinates for each vertex, sample from a texture image and
# apply the Phong lighting model
renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                    shader=HardFlatShader(cameras=cameras, lights=lights, device=device)).to(device)

texture = torch.reshape(texture, (texture.shape[0], texture.shape[1], out_res, out_res, 3)).to(device)
print("texture vector: {}".format(texture.shape))
mesh.textures = TexturesAtlas(texture)

meshes = mesh.extend(batch_size).to(device)
fake_images = renderer(meshes)


# mesh.textures = TexturesVertex(verts_features=texture)
# uniform_color = torch.ones_like(verts)[None, :, :3]  # Assuming verts is [V, 3]
# mesh.textures = TexturesVertex(verts_features=uniform_color.to(device))
# fake_images = renderer(mesh.to(device))


fake_batch = fake_images[..., :3].to(device).permute(0, 1, 2, 3).detach().cpu().numpy()
plt.figure(figsize=(10,10))
plt.imshow(fake_batch[0])
plt.axis("off")
plt.show()


