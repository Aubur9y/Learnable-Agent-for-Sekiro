import torch
torch.cuda.empty_cache()

foo = torch.tensor([1,2,3])
foo = foo.to('cuda')

print(foo)