from .base import GDMOptimizer
from .i_fgsm import IFGSM


# This is a different form of FGSM that we did not cover so I will keep my comments brief
# This seems to use momentum alongside normal IFGSM however it is not covered in the paper as extensively
# as the main DSA code.
class MIFGSM(IFGSM):
    def __init__(
        self,
        *,
        momentum=1,
        init_grad=0,
        **kwargs,
    ) -> None:
        self.momentum = momentum
        self.init_grad = init_grad
        super().__init__(**kwargs)

    def get_optimizer(self, stepsize: float):
        return GDMOptimizer(
            stepsize, self.momentum, self.distance.normalize, self.init_grad
        )
