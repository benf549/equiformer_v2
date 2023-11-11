import torch
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
# import pyg_lib 
# from torch_sparse import coalesce
# from torch_scatter import scatter
# from torch_cluster import radius

class DummyData():
    def __init__(self):

        self.natoms = (100,)
        self.pos = torch.randn(self.natoms[0], 3)
        self.batch = torch.zeros(self.natoms[0], dtype=torch.long)
        self.atomic_numbers = torch.randint(0, 20, (self.natoms[0],))


if __name__=="__main__":
    model = EquiformerV2_OC20(1, 1, 1)
    A = DummyData()
    model.forward(A)


