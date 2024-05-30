import os
import sys
import torch
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Add path for demo utils functions
from utility.plot_image_grid import image_grid 
from utility.utils import *

# Util function for loading meshes
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# Data structures and functions for rendering
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
    SoftPhongShader,
    TexturesVertex
)

# Utility Functions:

def dict_path(data_directory):
    # Map Object Locations
    id2ti = dict()
    for directory in os.listdir(data_directory):
        directory_path = os.path.join(data_directory, directory)
        textures = []
        models = []
        if os.path.isdir(directory_path):         
            for elem in os.listdir(directory_path+'/'+'models'):
                # TODO MODELS
                if elem.split('.')[-1] == 'obj':
                    models.append(directory_path+'/'+'models'+'/'+elem)
            id2ti[str(directory)] = models
    return id2ti


def multiview_rendering(
    mesh,
    num_views=20,
    parameters={
        "image_size": 128,
        "blur_radius": 0.0,
        "faces_per_pixel": 1,
        "bin_size": 32,  # Start with a moderate bin size
        "max_faces_per_bin": 50000  # Start with a high value
    },
    device='cpu'
):
    sigma = 1e-4
    result = {}

    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...])

    raster_settings = RasterizationSettings(
        image_size=parameters["image_size"],
        blur_radius=parameters["blur_radius"],
        faces_per_pixel=parameters["faces_per_pixel"],
        bin_size=parameters["bin_size"],  # Optional parameter
        max_faces_per_bin=parameters["max_faces_per_bin"]  # Optional parameter
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=camera,
            lights=lights
        )
    )

    meshes = mesh.extend(num_views)

    target_images = renderer(meshes, cameras=cameras, lights=lights)
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], T=T[None, i, ...]) for i in range(num_views)]

    raster_settings_silhouette = RasterizationSettings(
        image_size=128,
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=50,
        bin_size=parameters["bin_size"],  # Optional parameter
        max_faces_per_bin=parameters["max_faces_per_bin"]  # Optional parameter
    )
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )

    silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
    target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]

    raster_settings_soft = RasterizationSettings(
        image_size=128,
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=50,
        perspective_correct=False,
        bin_size=parameters["bin_size"],  # Optional parameter
        max_faces_per_bin=parameters["max_faces_per_bin"]  # Optional parameter
    )
    renderer_textured = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(device=device, cameras=camera, lights=lights)
    )

    result["camera"] = camera
    result["lights"] = lights
    result["target_cameras"] = target_cameras
    result["target_images"] = target_images
    result["target_rgb"] = target_rgb
    result["silhouette_images"] = silhouette_images
    result["target_silhouette"] = target_silhouette
    result["renderer_silhouette"] = renderer_silhouette
    result["renderer_textured"] = renderer_textured

    return result






def setup_renderer(device):

    raster_settings_soft = RasterizationSettings(
        image_size=128,
        blur_radius=np.log(1. / 1e-4 - 1.)*1e-4,
        faces_per_pixel=50,
        perspective_correct=False,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Differentiable soft renderer using per vertex RGB colors for texture
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=FoVPerspectiveCameras(device=device),
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=FoVPerspectiveCameras(device=device),
            lights=lights
        )
    )
    return renderer


#tmp functions
def visualize_prediction_simple(
        predicted_mesh,
        renderer,
        title='',
        silhouette=False
    ) -> None:

    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())
    plt.show()



def collect_data(meshes, device):
    # collect data
    all_data = []
    count = 0
    for mesh in tqdm(meshes, total=len(meshes), leave=True):
        count+=1
        if count==7: print()
        with torch.no_grad():
            results = multiview_rendering(mesh, device=device)
        tmp_dict = {
            "mesh": mesh,
            "lights": results["lights"],
            "camera": results["camera"],
            "target_cameras": results["target_cameras"],
            "target_images": results["target_images"],
            "target_rgb" : results["target_rgb"],
            "silhouette_images" : results["silhouette_images"],
            "target_silhouette" : results["target_silhouette"],
            "renderer_silhouette" : results["renderer_silhouette"],
            "renderer_textured" : results["renderer_textured"]
        }

        all_data.append(tmp_dict)
    return all_data


def visualize_collect_data(images):
    for i, elem in enumerate(images):
        plt.figure()
        plt.imshow(elem.cpu().numpy())
        plt.title(f"View {i}")
        plt.axis('off')
        plt.show()


# Load Object
def load_target_mesh(obj_location, device='cpu'):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Turn off default filtering
        try:
            mesh = load_objs_as_meshes([obj_location], device=device)
        except Warning as e:
            print("Warning eccolo:", e)

    if w:
        # print('sono finito qua, eh gia')
        verts_rgb = torch.ones_like(mesh.verts_packed())[None]  # (1, V, 3)
        mesh.textures = TexturesVertex(verts_features=verts_rgb.to(device))


    # We scale normalize and center the target mesh to fit in a sphere of radius 1
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
    # to its original center and scale.  Note that normalizing the target mesh,
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))
    return mesh


def plot_3d_mesh(mesh):
    # Render the plotly figure
    fig = plot_scene({
        "subplot1": {
            "mesh": mesh
        }
    })
    fig.show()


class Trainer():

    def __init__(self, device: str = 'cuda') -> None:
        self.device = device

    # Show a visualization comparing the rendered predicted mesh to the ground truth mesh
    def visualize_prediction(
        self,
        predicted_mesh,
        renderer,
        target_image,
        title='',
        silhouette=False
    ) -> None:

        inds = 3 if silhouette else range(3)
        with torch.no_grad():
            predicted_images = renderer(predicted_mesh)
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

        plt.subplot(1, 2, 2)
        plt.imshow(target_image.cpu().detach().numpy())
        plt.title(title)
        plt.axis("off")
        plt.show

    # Losses to smooth / regularize the mesh shape
    def update_mesh_shape_prior_losses(self, mesh, loss):
        loss["edge"] = mesh_edge_loss(mesh)
        loss["normal"] = mesh_normal_consistency(mesh)
        loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


    def update_current_loss(self, loss_dictionary,  losses, rgb_mode):
        sum_loss = torch.tensor(0.0, device=self.device)
        for k, l in loss_dictionary.items():
            if k == "rgb" and not rgb_mode:
                continue
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))
        return sum_loss
    
    

    def train_loop(
        self,
        losses,
        src_mesh,
        optimizer,
        deform_verts = None,
        sphere_verts_rgb = None,
        mesh_gen: torch.nn.Module = None,
        lights: list = [],
        target_rgb: list = [],
        target_cameras: list = [],
        target_silhouette: list = [],
        current_renderer: MeshRenderer = None,
        num_views: int = 20,
        num_iter: int = 1000,
        plot_period: int = 250,
        rgb_mode: bool = True,
        num_views_per_iteration: int = 5,
    ) -> pytorch3d.structures.meshes.Meshes:

        new_src_mesh = src_mesh
        loop = tqdm(range(num_iter))

        for i in loop:

            # Initialize optimizer
            if mesh_gen:
                mesh_gen.train()
            optimizer.zero_grad()

            # deform mesh with model
            if mesh_gen:
                # print(f"input shape: {new_src_mesh.verts_packed().shape}") # debug
                model_input = src_mesh.verts_packed().to(self.device)
                deform_verts = mesh_gen(model_input)
                new_src_mesh = src_mesh.offset_verts(deform_verts) 

            # deform the vertex matrix directly 
            else:
                new_src_mesh = mesh_gen
                new_src_mesh = src_mesh.offset_verts(deform_verts)

            if rgb_mode:
                new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)

            # Losses to smooth /regularize the mesh shape
            loss = {k: torch.tensor(0.0, device=self.device) for k in losses}
            self.update_mesh_shape_prior_losses(new_src_mesh, loss)

            # Compute the average silhouette loss over two random views, as the average
            # squared L2 distance between the predicted silhouette and the target
            # silhouette from our dataset
            #TEST
            # total_silhouette_loss = torch.tensor(0.0, device=self.device)
            # total_rgb_loss = torch.tensor(0.0, device=self.device) if rgb_mode else None

            for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
                images_predicted = current_renderer(new_src_mesh, cameras=target_cameras[j], lights=lights)
                predicted_silhouette = images_predicted[..., 3]
                loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
                # total_silhouette_loss += loss_silhouette / num_views_per_iteration
                loss["silhouette"] += loss_silhouette / num_views_per_iteration

                if rgb_mode:
                    # Squared L2 distance between the predicted RGB image and the target
                    # image from our dataset
                    predicted_rgb = images_predicted[..., :3]
                    loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
                    loss["rgb"] += loss_rgb / num_views_per_iteration
                    # total_rgb_loss += loss_rgb / num_views_per_iteration #TEST

            # TEST
            # loss["silhouette"] += total_silhouette_loss
            # if rgb_mode:
            #     loss["rgb"] += total_rgb_loss        




            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=self.device)
            for k, l in loss.items():
                if k == "rgb" and not rgb_mode: continue
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))

            # sum_loss = self.update_loss(
            #     new_src_mesh,
            #     loss,
            #     losses,
            #     lights=lights,
            #     renderer=current_renderer,
            #     target_cameras=target_cameras,
            #     num_views=num_views,
            #     views_per_iteration=num_views_per_iteration,
            #     target_silhouett=target_silhouette
            # )

            # Print the losses
            loop.set_description("total_loss = %.6f" % sum_loss)

            # Plot mesh
            if i % plot_period == 0:
                if rgb_mode:
                    self.visualize_prediction(
                        new_src_mesh,
                        title="iter: %d" % i,
                        silhouette=False,
                        renderer=current_renderer,
                        target_image=target_rgb[1]

                    )
                else:
                    self.visualize_prediction(
                        new_src_mesh,
                        title="iter: %d" % i,
                        silhouette=True,
                        renderer=current_renderer,
                        target_image=target_silhouette[1]
                    )

            # Optimization step
            sum_loss.backward()
            optimizer.step()
         
        return new_src_mesh
    



    # Plot losses as a function of optimization iteration
    def plot_losses(self, losses):
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        for k, l in losses.items():
            ax.plot(l['values'], label=k + " loss")
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Loss vs iterations", fontsize="16")
