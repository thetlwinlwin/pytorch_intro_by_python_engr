import torch

""" This will give AttributeError """


FILE = "./whole_model.pt"
load_model = torch.load(FILE)
load_model.eval()

for i in load_model.parameters():
    print(i)
