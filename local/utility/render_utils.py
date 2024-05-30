import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex
)

import os
import sys
sys.path.append(os.path.abspath(''))


from local.utils.plot_image_grid import image_grid


def load_target_mesh(data_dir, filename, device):
    mesh = load_objs_as_meshes([filename],device=device)
    #normalize the meshes loaded
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_(1.0 / float(scale))
    return mesh

def multi_rgb_images(num_views,mesh):
    #Create batch of views
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    #Place a point light in front of object. UNDERSTAND BETTER THIS.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    #choose one particular view
    camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...])

    #Define setting for rasterizzation and shading
    raster_settings = RasterizationSettings(
        image_size = 128,
        blur_radius= 0.0, 
        faces_per_pixel= 1,
    )

    #Create a Phong rendered by composing a rasterizer and a shader
    renderer = MeshRenderer(
        rasterizer = MeshRasterizer(
            cameras = camera,
            raster_settings = raster_settings
        ),
        shader = SoftPhongShader(
            device = device,
            cameras = camera,
            lights = lights
        )
    )

    #Create a batch of meshes by repeting mesh and associated textures.
    meshes = mesh.extend(num_views)

    #Render the mesh from each p.o.w
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], 
                                            T=T[None, i, ...]) for i in range(num_views)]
    
    return meshes, target_rgb, target_images, camera, cameras, lights

def multi_silhouette_images(camera,cameras,sigma=1e-4):
   
    # Rasterization settings for silhouette rendering  
    raster_settings_silhouette  = RasterizationSettings(
        image_size=128,
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=50,
    )
    # Silhouette renderer 
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )

    # Render silhouette images.  The 3rd channel of the rendering output is 
    # the alpha/silhouette channel
    silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
    target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]

    return silhouette_images, target_silhouette

'''
MAIN
'''
if __name__ == '__main__':

    #SETUP
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")


    #CHANGE HERE TO GENERALIZE THE METHOD 
    DATA_DIR = "./data" 
    obj_path = "cow_mesh/cow.obj" 
    obj_filename = os.path.join(DATA_DIR,obj_path)

    mesh = load_target_mesh(DATA_DIR,obj_filename,device)

    #DATASET RGB-MULTIVIEW CREATION
    num_views = 20

    #FIX: MIGLIORARE LEGGIBILITA' CREANDO UNA CLASSE CONTENENTE I PARAMETRI?
    meshes, target_rgb, target_images, camera, cameras, lights = multi_rgb_images(num_views,mesh)


    # RGB IMAGES
    image_grid(target_images.cpu().numpy(), rows=4, cols=5, rgb=True)
    plt.show()



#DATASET RGB-SILHOUETTE CREATION

silhouette_images, target_silhouette = multi_silhouette_images(camera,cameras)

#SILHOUETTE IMAGES
image_grid(silhouette_images.cpu().numpy(), rows=4, cols=5, rgb=False)
plt.show()
