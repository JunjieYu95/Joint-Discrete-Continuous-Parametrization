import torch
from conditionalVAE.models import cVAE
from conditionalVAE.training import Trainer
from utils.dataloaders import get_straight_geological_scenario_dataloaders
from viz.visualize import Visualizer
from torch import optim


batch_size = 64
lr = 5e-4
epochs = 30

# Check for cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('gpu is activated')

# Load data
train_loader, test_loader = get_straight_geological_scenario_dataloaders(batch_size=batch_size)
img_size = (1, 64, 64)

# Define latent spec and model
latent_spec = {'cont': 10, 'cond': 4}
model = cVAE(img_size=img_size, latent_spec=latent_spec,
            use_cuda=use_cuda)
if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# visualier
viz = Visualizer(model, include_label = True)


# Define trainer
log_folder = 'Experiments/model_training/straight_generative_scenario/conditional_VAE'
trainer = Trainer(model, optimizer,log_folder = log_folder,
                  cont_capacity=[0.0, 0, 25000, 10],
                  use_cuda=use_cuda)

# Train model for 100 epochs
trainer.train(train_loader, test_loader, epochs=epochs,monitor_interval = 10, 
              save_training_gif=(None, viz))

# Save trained model
torch.save(trainer.model.state_dict(), 'model.pt')
