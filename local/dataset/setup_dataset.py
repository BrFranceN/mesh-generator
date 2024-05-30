"""
This script implements the Multi View Dataset class used to train the mesh generator network
"""

import os
import sys

# add project root to sys path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)\

from utility.utils import *
import mv_dataset
# from mv_dataset import MultiViewDataset

if __name__ == '__main__':
    
    # Cuda Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print("device: {}".format(device))
   
    
    id2paths = dict_path(current_dir + "/../../data/dataset-03797390-20/")

    # make dataset
    meshes = [load_target_mesh(id2paths[key][0], device=device) for key in id2paths]
    all_data = collect_data(meshes, device=device)

    # Data exploration:
    # for elem in all_data:
    #     plot_3d_mesh(elem["mesh"])
    # visualize_collect_data(all_data[0]['target_images'])

    dataset_instance = mv_dataset.MultiViewDataset(all_data)

    torch.save(dataset_instance, f"{current_dir}/multiview_dataset.pt")