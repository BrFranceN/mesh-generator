# import os
# import sys
# # add project root to sys path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(parent_dir)\

# from utility.utils import *
# import dataset.mv_dataset
# from dataset.mv_dataset import MultiViewDataset


#test
import os
import sys
import torch
torch.autograd.set_detect_anomaly(True)

# Torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# add project root to sys path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Ensure the dataset directory is in the path
dataset_dir = os.path.join(parent_dir, 'dataset')
sys.path.append(dataset_dir)

from utility.utils import *
from mv_dataset import MultiViewDataset  # Ensure this path is correct


class SimpleVertexDeformer(nn.Module):

    def __init__(self, in_features: int = 3, hidden_dim: int = 256, out_features: int = 3) -> None:
        super(SimpleVertexDeformer, self).__init__()

        # layers:
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.batchnorm1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(in_features)
        self.fc3 = nn.Linear(hidden_dim, out_features)

    def forward(self, input):
        # output1 = self.relu(self.batchnorm1(self.fc1(input)))
        # output2 = self.relu(self.batchnorm2(self.fc2(output1)))
        output1 = self.relu(self.fc1(input))
        output2 = self.relu(self.fc2(output1))
        output3 = self.fc3(output2)
        return output3


###############################
class DeformNet(nn.Module):
  def __init__(self, verts_dim=3, hidden_dim=256, disp_dim=3, norm_ratio=0.1):
    super(DeformNet, self).__init__()

    self.l1 = nn.Linear(verts_dim,hidden_dim)
    self.a1 = nn.ReLU()
    self.l2 = nn.Linear(hidden_dim,hidden_dim)
    self.a2 = nn.ReLU()
    self.l3 = nn.Linear(hidden_dim,disp_dim)
    self.norm_ratio = norm_ratio

  def forward(self,x):
    x = self.l1(x)
    x = self.a1(x)
    x = self.l2(x)
    x = self.a2(x)
    x = self.l3(x) * self.norm_ratio

    return x
################################



if __name__ == '__main__':
    

    # argv 
    load_dataset_from_file = False
    inference = False

    # Cuda Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print("device: {}".format(device))


    if load_dataset_from_file  and os.path.exists(current_dir + "/../dataset/multiview_dataset.pt"):
        multiview_dataset = torch.load(current_dir + "/../dataset/multiview_dataset.pt")

    else:    
        # id2paths = dict_path(current_dir + "/../../data/dataset-03797390-20/")
        id2paths = dict_path(current_dir + "/../../data/dataset_cow/")
        meshes = [load_target_mesh(id2paths[key][0], device=device) for key in id2paths]
        all_data = collect_data(meshes, device=device)
        multiview_dataset = MultiViewDataset(all_data)
        
    print(f"id2paths:{id2paths} ")
    print(f"Multi view dataset loaded \tnum items: {len(multiview_dataset)}")


    random_sample = 0
    if not inference:
        # define source mesh
        ico_mesh = ico_sphere(4, device)
        verts_shape = ico_mesh.verts_packed().shape
        deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True) # optional
        sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True) #optional

        # set texture (for now white!)
        ico_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)

        # parameters
        curr_obj_params = multiview_dataset.get_parameters(random_sample)
        lights = curr_obj_params["lights"]
        camera = curr_obj_params["camera"]
        target_rgb = curr_obj_params["target_rgb"]
        target_images = curr_obj_params["target_images"]
        target_cameras = curr_obj_params["target_cameras"]
        target_silhouette = curr_obj_params["target_silhouette"]
        silhouette_images = curr_obj_params["silhouette_images"]
        obj_renderer = curr_obj_params["renderer"]
        num_views = len(target_rgb)

        # visualize multiview data
        # RGB images
        # image_grid(target_images.cpu().numpy(), rows=4, cols=5, rgb=True)
        # plt.show()

        # Visualize silhouette images
        # image_grid(silhouette_images.cpu().numpy(), rows=4, cols=5, rgb=False)
        # plt.show()
        
        
        # model
        mesh_gen = SimpleVertexDeformer(in_features=verts_shape[1], out_features=verts_shape[1]).to(device)
        experimental_model = DeformNet(verts_dim=3, hidden_dim=256, disp_dim=3, norm_ratio=0.1).to(device)
        # training
        trainer = Trainer(device=device)
        optimizer = torch.optim.SGD(mesh_gen.parameters(), lr=1.0, momentum=0.9)
        # optimizer = torch.optim.AdamW(mesh_gen.parameters(), lr=0.1)
        losses = {"rgb": {"weight": 1.0, "values": []},
                "silhouette": {"weight": 1.0, "values": []},
                "edge": {"weight": 1.0, "values": []},
                "normal": {"weight": 0.01, "values": []},
                "laplacian": {"weight": 1.0, "values": []},
                }
        

        new_src_mesh = trainer.train_loop(
            losses=losses,
            src_mesh=ico_mesh,
            optimizer=optimizer,
            num_views=num_views,
            deform_verts=deform_verts,
            sphere_verts_rgb=sphere_verts_rgb,
            mesh_gen=None, #experimental_model,#CHANGE
            lights=lights,
            target_rgb=target_rgb,
            target_cameras=target_cameras,
            target_silhouette=target_silhouette,
            current_renderer=obj_renderer,
            rgb_mode=False
        )
        trainer.plot_losses(losses)

        torch.save(mesh_gen, f"{current_dir}/checkpoint.pt")

    
    #inference
    if os.path.exists('checkpoint.pt'):
        print("inference mode:")
        mesh_model_test = torch.load("checkpoint.pt")
        test_mesh = ico_sphere(4,device)
        deformed_verts_test = mesh_model_test(test_mesh.verts_packed())
        test_mesh = test_mesh.offset_verts(deformed_verts_test)
        plot_3d_mesh(test_mesh)
        plot_3d_mesh(multiview_dataset[random_sample][0])

        #test
        


    # TODO test inference






