import numpy as np
import torch as tc


x = tc.rand(4, )
y = tc.randn(4, 4)
print(x.matmul(y))