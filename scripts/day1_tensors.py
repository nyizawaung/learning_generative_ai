import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


x = torch.tensor([1,2,3], device = device)
print("x:\n", x)

y = torch.tensor([[1,2], [3,4], [5,6]], device = device)
print("y:\n", y)

z = torch.zeros((4,4))
print("z:\n", z)

r = torch.rand((2,2))
print("r:\n", r)

a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
print("a + b:\n", a + b)

m1 = torch.tensor([[1,2],[3,4]])
m2 = torch.tensor([[5,6],[7,8]])
print("matrix product: \n", torch.matmul(m1,m2))

reshaped = torch.arange(25).reshape((5,5))
#reshaped = torch.arange(3,4).reshape((12))
print("reshaped:\n", reshaped)


t = torch.ones(2,2)
t_gpu = t.to(device)
print("Tensor on: ", t_gpu.device)