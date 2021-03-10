import torch.nn as nn
import torch
N, D_in, H, D_out = 64, 1000, 100, 10
#
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
q = 2
class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # define the model architecture
        self.linear1 = torch.nn.Linear(D_in, H, bias=False)
        self.linear2 = torch.nn.Linear(H, D_out, bias=False)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred











model = Net(D_in, H, D_out)
loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for it in range(500): # Forward pass
    optimizer.zero_grad()
    y_pred = model(x) # model.forward()
    # compute loss
    loss = loss_fn(y_pred, y) # computation graph
    fair_loss = (1/(1+q)) * pow(loss, (q + 1))
    print(it, loss.item())
    print("-----------")
    print("the fair loss is:",it, fair_loss.item())
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # update model parameters
    optimizer.step()
