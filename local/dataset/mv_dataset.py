import os
import sys
import torch
from torch.utils.data import Dataset

# add project root to sys path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utility.utils import *


class MultiViewDataset(Dataset):

    def __init__(self, data, device='cuda') -> None:
        super(MultiViewDataset, self).__init__()

        # data
        self.device = device
        self.data = data
        self.meshes = [sample["mesh"] for sample in self.data]
        self.target_cameras = [sample["target_cameras"] for sample in self.data]
        self.target_rgb = [sample["target_rgb"] for sample in self.data]
        self.lights = [sample["lights"] for sample in self.data]
       
        # self.target_images = [sample["target_images"] for sample in self.data]

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        return self.meshes[idx], self.target_cameras[idx], self.target_rgb[idx], self.lights[idx]

    def get_parameters(self, idx):
        return self.data[idx]
    
    # def custom_collate(self, batch):
    #     # Assuming each item in batch is a tuple like (mesh, other_data)


    #     ico_mesh = ico_sphere(4, self.device).to(self.device) # fixed icosphere
    #     verts_shape = ico_mesh.verts_packed().shape
    #     sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=self.device, requires_grad=True)
    #     ico_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)
        
    #     # meshes = [item[0] for item in batch]
    #     meshes = [ico_mesh for item in batch]
    #     other_data = [item[1:] for item in batch]
    #     target_cameras = [item[1] for item in batch]
    #     target_rgb = [item[2] for item in batch]
    #     lights = [item[3] for item in batch]  
        
    #     # Convert the meshes into a batched Meshes object
    #     verts = [mesh.verts_packed() for mesh in meshes]
    #     faces = [mesh.faces_packed() for mesh in meshes]
    #     batch_meshes = Meshes(verts=verts, faces=faces)
        
    #     # Use the default collate function for the rest of the data
    #     # other_data_collated = torch.utils.data.default_collate(other_data)
        
    #     # Return the collated batch as a tuple (batch_meshes, other_data_collated)
    #     return batch_meshes, (target_cameras, target_rgb, lights) #other_data #other_data_collated