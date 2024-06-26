from torch.utils.cpp_extension import load
import os

ca_sa = load(name="ca_sa", sources=os.path.join(os.path.dirname(__file__), "ca_sa.cpp"))

__all__ = ['ca_sa']
