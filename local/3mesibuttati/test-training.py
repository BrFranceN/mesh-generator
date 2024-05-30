import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

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

from tqdm import tqdm

from data.dataset import MVDataset
from data.dataset import mv_transform
from models.gcn import GCN
from models.discriminator import Discriminator
from utils.mesh_utils import sparse_mx_to_torch_sparse_tensor
from utils.mesh_utils import mesh_features, mesh_features_dual

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# data
train_set = MVDataset(images_dir='data/train/images', transform=mv_transform)


# networks parameters
in_features = 6
n_hidden = 64
out_features = 3
noise_dim = 8
dropout = 0.5

# networks definition
netD = Discriminator()
netG = GCN(in_features=in_features, n_hidden=n_hidden, out_features=out_features,
               noise_dim=noise_dim, dropout=dropout).to(device)

# training parameters
real_label = 1.
fake_label = 0.

num_epochs = 1
batch_size = 1

beta1 =  0.5,
beta2 = 0.999,

lr = 0.001
weight_decay = 5e-4

criterion = torch.nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, weight_decay=weight_decay)

# loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
total_models = len(train_loader.dataset)

# save data for stats
loss_G = []
loss_D = []

# output parametrs 
fixed_elevation = torch.FloatTensor([41.5770])
fixed_azimuth = torch.FloatTensor([33.2110])
image_size = 6
out_res = 1

print("Training...")
for epoch in range(num_epochs):
    # for each epoch iterate over the obj files and train
    print(f"Epoch: {epoch + 1} / {num_epochs}")
    errorD_total = 0.0
    errorG_total = 0.0

    # set both models to train mode
    netG.train()
    netD.train()

    for batch in train_loader:
        images, rotations, transform_matrices = batch

        obj_path = 'data/train/chair.obj'
        verts, faces, aux = load_obj(obj_path, device=device)
        mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])

        features, adj = mesh_features_dual(obj_path='dataset', mesh=mesh)
        features = features.to(device)
        adj = adj.to(device)

        ################## TODO: get R and t from transform file ################
        # sample new camera location
        distance = 1.0
        elevation = fixed_elevation #torch.FloatTensor(batch_size).uniform_(0, 180)
        azimuth = fixed_azimuth #torch.FloatTensor(batch_size).uniform_(-180, 180)

        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        #########################################################################
        ###################### TODO: new version  ###############################
        R = rotations
        T = transform_matrices


        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                                cull_backfaces=True)

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

        # render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
        

        # zero grad the netD
        netD.zero_grad()
        # extract RGB images from these real rendered images
        real_batch = images.to(device).permute(0, 3, 1, 2)
        
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        # forward pass the real batch through D
        output = netD(real_batch).view(-1)
        errD_real = criterion(output, label)
        # calculate gradients for real batch
        errD_real.backward()

        # generate noise
        noise = torch.randn(noise_dim).to(device)
        texture = netG(features, adj, noise).to(device).unsqueeze(dim=0)
     
        texture = torch.reshape(texture, (texture.shape[0],
                                            texture.shape[1],
                                            out_res, out_res,
                                            3)).to(device)
        mesh.textures = TexturesAtlas(texture)
   

        # will produce fake images of shape [batch_sizeximage_sizeximage_sizex4(RGBA)]
        meshes = mesh.extend(batch_size).to(device)
        fake_images = renderer(meshes)
        fake_batch = fake_images[..., :3].to(device).permute(0, 3, 1, 2)

        label.fill_(fake_label)
        # pass fake batch to netD
        output = netD(fake_batch.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()

        # update the discriminator
        optimizerD.step()

        errD = errD_real.item() + errD_fake.item()

        errorD_total += errD

        # Train generator
        # zero grad the generator
        netG.zero_grad()


        errG = -torch.mean(netD(fake_batch))

        # calculate gradients
        # use_image_loss:
        errG += torch.mean(torch.abs(real_batch - fake_batch))

        errG.backward()

        # update generator
        optimizerG.step()

        # some book keepingß
        errorG_total += errG.item()

    # Output training stats
    errorD_total = errorD_total / total_models
    errorG_total = errorG_total / total_models

    loss_D.append(errorD_total)
    loss_G.append(errorG_total)
    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch + 1, num_epochs, errorD_total, errorG_total))






# plot losses
fig = plt.figure(figsize=(10, 5))
plt.title("Loss During Training")
plt.plot(loss_G, label="G")
plt.plot(loss_D, label="D")

plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
# plt.savefig(save_folder + "/" + "loss.png")
plt.close(fig)