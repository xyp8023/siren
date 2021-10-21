# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules, diff_operators
import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')
p.add_argument('--hidden_features', type=int, default=128,
               help='Width of the hidden layers of the MLP.')
p.add_argument('--num_hidden_layers', type=int, default=3,
               help='Number of the hidden layers of the MLP.')

p.add_argument('--checkpoint_path', default="checkpoints/model_current.pth", help='Checkpoint to trained model.')
opt = p.parse_args()
root_path = os.path.join(opt.logging_root, opt.experiment_name)
#img_dataset = dataio.Camera()# camera man
img_dataset = dataio.NumpyFile("data/heightmap_square.npy") # bathymetry
#coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
coord_dataset = dataio.Implicit2DWrapperFromArray(img_dataset, sidelength=480, compute_diff='all')
#image_resolution = (512, 512)
image_resolution = (480, 480)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
        or opt.model_type == 'softplus':
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', sidelength=image_resolution, hidden_features=opt.hidden_features, num_hidden_layers=opt.num_hidden_layers)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=image_resolution)
else:
    raise NotImplementedError

model.load_state_dict(torch.load(os.path.join(root_path, opt.checkpoint_path)))
model.cuda()

coords = coord_dataset.__getitem__(0)[0]["coords"]
print(type(coords)) # tensor
print(coords.shape) # (N, 2)
for (model_input, gt) in dataloader:
    #print(model_input["coords"].shape)
    model_input = {key: value.cuda() for key, value in model_input.items()}
    model_output = model(model_input)
    model_out = model_output["model_out"] # (1, N, 1)
    gt_img = gt['img'].view(image_resolution).cpu().detach().numpy()
pred_img = model_out.view(image_resolution)
img_gradient = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
pred_grad = dataio.grads2img(dataio.lin2img(img_gradient))

pred_grad = pred_grad.cpu().detach().numpy() #(3,480,480)
pred_grad = np.swapaxes(np.swapaxes(pred_grad,0,1), 1,2)
print("pred_grad: ", pred_grad.shape) # (480,480,3)

pred_img = pred_img.cpu().detach().numpy()


print("pred_img: ", pred_img.shape)
print("gt_img: ", gt_img.shape)
#def create_img(model, filename, resolution):
#for key in model_out.keys():
#    print(key)
plt.figure()
plt.imshow(pred_img, cmap='turbo')
plt.title("pred_img")
print("pred_img max: ", pred_img.max(), " pred_img min: ", pred_img.min())

plt.figure()
plt.imshow(gt_img, cmap='turbo')
plt.title("gt_img")
print("gt_img max: ", gt_img.max(), " gt_img min: ", gt_img.min())


plt.figure()
plt.imshow(pred_grad)
plt.title("pred_grad")
print("pred_grad max: ", pred_grad.max(), " pred_grad min: ", pred_grad.min())


plt.show()
#create_img(model, os.path.join(root_path, 'test'), image_resolution)
