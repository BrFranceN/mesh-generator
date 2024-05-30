import os
import sys
# add project root to sys path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)\

from utility.utils import *




# DEFORMER EXPERIMENT
class VertexDeformer(nn.LightningModule):
    def __init__(self, input_features, output_features, renderer):
        super(VertexDeformer, self).__init__()
        self.fc1 = nn.Linear(input_features, output_features)
        self.batchnorm1 = nn.BatchNorm1d(output_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_features, input_features)
        self.batchnorm2 = nn.BatchNorm1d(input_features)
        self.renderer = renderer
        self.reference_ico_mesh = ico_sphere(4, self.device).to(self.device)
        verts_shape = self.reference_ico_mesh.verts_packed().shape
        sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)
        self.reference_ico_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)


    def forward(self, x):
        x = self.relu(self.batchnorm1(self.fc1(x)))
        x = self.batchnorm2(self.fc2(x))
        return x

    def training_step(self, batch, batch_idx):
        verts = batch[0].verts_packed()  # ichospheres
        verts_flat = verts.view(-1, 3) # flatten verts for input to the linear layer         
        deformed_vertices = self.forward(verts_flat).to(self.device)

        print(f"batch 0 -> {type(batch[0])}")
        print(f"batch 1 -> {type(batch[1])}")
        print(f"len list  -> {len(batch[1])}")
        print(f"contain type list  -> {len(batch[1][0])}")
        print(f"element 1  -> {type(batch[1][0][0])}")
        print(f"element 2 -> {type(batch[1][0][1])}")
        print(f"element 3  -> {type(batch[1][0][2])}")
        print(f"element 1 len  -> {len(batch[1][0][0])}")
        print(f"element 2 len  -> {len(batch[1][0][1])}")



        # target_cameras = batch[1][0]
        # target_rgb = batch[1][1]
        # lights = batch[1][2]


        target_cameras, target_rgb, lights = batch[1]
        deformed_meshes = Meshes(verts=[deformed_vertices.to(self.device)], faces=[self.reference_ico_mesh.faces_packed().to(self.device)]).to(self.device)
        #set texture to deformed_meshes:
        verts_shape = deformed_meshes.verts_packed().shape
        sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=self.device, requires_grad=True)
        deformed_meshes.textures = TexturesVertex(verts_features=sphere_verts_rgb)     
         
        loss = self.compute_loss(deformed_meshes, target_cameras, lights, target_rgb)
        self.log('train_loss', loss)
        # loss computation
        # loss = self.compute_loss(deformed_vertices, target_cameras, lights, target_rgb)
        # self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # verts = batch[0].verts_packed()  # Shape: (N, 3)
        # verts_flat = verts.view(-1, 3) # flatten verts for input to the linear layer 
        # deformed_vertices = self.forward(verts_flat)

        # target_cameras = batch[1][0]
        # target_rgb = batch[1][1]
        # lights = batch[1][2]
        # target_cameras, target_rgb, lights = batch[1]
        # deformed_meshes = Meshes(verts=[deformed_vertices], faces=[self.ico_mesh.faces_packed()])        
        # loss = self.compute_loss(deformed_meshes, target_cameras, lights, target_rgb)
        # deformed_meshes = Meshes(verts=[deformed_vertices], faces=[self.ico_mesh.faces_packed()])
     
        # loss = self.compute_loss(deformed_meshes, target_cameras, lights, target_rgb)
        # self.log('val_loss', loss)

        #TEST
        verts = batch[0].verts_packed()  # Extract vertices from batched meshes
        verts_flat = verts.view(-1, 3)  # Flatten verts for input to the linear layer
        deformed_vertices = self.forward(verts_flat).to(self.device)

        target_cameras, target_rgb, lights = batch[1]
        deformed_meshes = Meshes(verts=[deformed_vertices], faces=[self.reference_ico_mesh.faces_packed()]).to(self.device)

        #set texture to deformed_meshes:
        verts_shape = deformed_meshes.verts_packed().shape
        sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=self.device, requires_grad=True)
        deformed_meshes.textures = TexturesVertex(verts_features=sphere_verts_rgb)
        

        loss = self.compute_loss(deformed_meshes, target_cameras, lights, target_rgb)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        # Configure optimizers and optionally learning rate schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def compute_loss(self, deformed_meshes, target_cameras, lights, target_rgb):
        #test
        images_predicted = self.renderer(deformed_meshes, cameras=target_cameras[:][1][0], lights=lights)
        predicted_rgb = images_predicted[..., :3]
        return torch.nn.functional.mse_loss(predicted_rgb, target_rgb[:][1][0])
        #finish test


        # Compute some loss function here
        # for j in np.random.permutation(2).tolist()[:self.num_views_per_iteration]:
        images_predicted = self.renderer(deformed_vertices, cameras=target_cameras[:][1], lights=lights[:])
        predicted_rgb = images_predicted[..., :3]
        # tasrgets = target_rgb[1]
            
        return torch.nn.functional.mse_loss(predicted_rgb, target_rgb[:][1])

# Cuda Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")

    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print("device: {}".format(device))




    
# MAIN LOGICA CODICE
id2paths = dict_path("../../data/dataset-03797390-20/")


# make dataset
meshes = [load_target_mesh(id2paths[key][0], device=device) for key in id2paths]
all_data = collect_data(meshes, device=device)

# Data exploration:
# for elem in all_data:
#     plot_3d_mesh(elem["mesh"])
# visualize_collect_data(all_data[0]['target_images'])

mv_dataset = MultiViewDataset(all_data)
# mv_dataloader = DataLoader(mv_dataset, collate_fn=mv_dataset.custom_collate, batch_size=8)
mv_dataloader = DataLoader(mv_dataset, collate_fn=mv_dataset.custom_collate, batch_size=8)

print(f"batch size : {len(mv_dataloader)}")

renderer = setup_renderer(device)


# Initialize the deformer network
ico_mesh = ico_sphere(4, device)
verts_shape = ico_mesh.verts_packed().shape
sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)


vertex_deformer = VertexDeformer(3, 3,renderer).to(device)  # 3 input features (x, y, z) per vertex
print(f"model device: {vertex_deformer.device}")


# define trainer
# TODO: find best learning rate
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model=vertex_deformer, train_dataloaders=mv_dataloader)


# Get the original vertices of the mesh
original_vertices = ico_mesh.verts_packed().to(device)  # Shape [num_vertices, 3]
# Pass the original vertices through the network to get new vertices
vertex_deformer = vertex_deformer.cuda()
print(f"model device: {vertex_deformer.device}")
print(f"ico device: {original_vertices.device}")

deformed_vertices = vertex_deformer.forward(original_vertices)

# Set all vertex colors to white and reshape
vertex_colors = torch.zeros([1, deformed_vertices.shape[0], 3], device=device)  # Shape [1, num_vertices, 3]
textures = TexturesVertex(verts_features=vertex_colors)
deformed_mesh = Meshes(verts=[deformed_vertices], faces=ico_mesh.faces_list(), textures=textures)

print(deformed_mesh)

# Create a new mesh with the deformed vertices and original faces

renderer = setup_renderer(device = device)

list_mesh = [deformed_mesh]

# visualize_prediction(deformed_mesh, renderer)

all_data = collect_data(list_mesh,device=device)

# visualize_collect_data(all_data[0]['target_images'])
# plot_3d_mesh(all_data[0]["mesh"])






