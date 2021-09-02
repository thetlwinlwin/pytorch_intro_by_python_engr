import torch
import torch.nn as nn
from torch.optim import optimizer

# we have to rebuild the saved model and call that model.


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features=n_input_features, out_features=1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


load_state_dict_to_model = Model(n_input_features=6)
FILE = "state_dict.pt"
load_state_dict_to_model.load_state_dict(torch.load(FILE))
load_state_dict_to_model.eval()

for param in load_state_dict_to_model.parameters():
    print(param)

loaded_model = Model(n_input_features=6)
# initially we set the lr = 0.1
optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.1)

loaded_checkpoint = torch.load("checkpoint.pt")

loaded_model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

# now the learning rate is 0.01 as we loaded from the checkpoint.pt
print(optimizer.state_dict())
