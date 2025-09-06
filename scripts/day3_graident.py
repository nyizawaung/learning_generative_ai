import torch


# parameter we want to learn
w = torch.tensor(1.0, requires_grad=True)

# input and target
x = torch.tensor(2.0)
target = torch.tensor(10.0)


#forward pass

y = w* x
loss = (y - target) ** 2 #mesan squared error

print("y:", y.item())
print("loss:", loss.item())

#backward pass
loss.backward()
print("Gradient of loss w.r.t w:", w.grad.item())

learning_rate = 0.1
w.data = w.data - learning_rate * w.grad
print("Updated w:", w.item())