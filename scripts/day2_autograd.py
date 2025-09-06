import torch
x = torch.tensor(2.0, requires_grad = True)
y = x**2 + 3*x + 1
print("y:\n", y)

y.backward()
print("x:", x)
print("dy/dx:", x.grad) # derivative of x^2 + 3x + 1 = 2x + 3 = 7


a = torch.tensor(3.0, requires_grad = True)
b = torch.tensor(4.0, requires_grad  = True)
f = a* b + a**3
f.backward()

print("df/da:", a.grad) # derivative w.r.t a-> b + 2a
print("df/db:", b.grad) # derivative w.r.t b-> a


w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor (2.0)
target = torch.tensor(10.0)

#forward
y = w * x
loss = (y - target) ** 2 #MSE

#backward
loss.backward()

print("w => ",w)
print("Gradient of loss w.r.t w:", w.grad) # 2*(wx - target)*x = 2*(2 - 10)*2 = -32