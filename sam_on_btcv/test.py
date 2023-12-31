import torch
import functools

class A:
    def __init__(self):
        self.a = torch.tensor(1.,requires_grad=True)

    @torch.no_grad()
    def forawrd(self):
        self.b = self.a ** 2
        print(self.b)
        self.b.backward()
        
class B(A):
    def __init__(self):
        super().__init__()
    
    # @functools.wraps(A.forawrd)    
    def forward(self):
        # super().forawrd()
        self.b = self.a ** 2
        print(self.b)
        self.b.backward()
        
# a = A()
# a.forawrd()
b = B()
b.forawrd()