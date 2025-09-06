import torch


# parameter to learn
w = torch.tensor(1.0, requires_grad=True)

#input and target
x = torch.tensor(2.0)
target = torch.tensor(10.0)

learning_rate = 0.1


for epoch in range(20):

    y = w * x
    loss = (y - target) ** 2 #MSE

    #backward pass
    loss.backward()

    with torch.no_grad():
        #print("Gradient of loss w.r.t w:", w.grad) # 2*(wx - target)*x = 2*(2 - 10)*2 = -32
        w -= learning_rate * w.grad

    w.grad.zero_() # reset gradients to zero (!important)

    print(f"Epoch {epoch+1:2d}: w = {w.item():.4f}, y = {y.item():.4f}, loss = {loss.item():.4f}")
    