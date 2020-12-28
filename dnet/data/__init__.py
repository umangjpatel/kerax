from typing import Tuple

from dnet.utils.tensor import Tensor


class Dataloader:

    def __init__(self, train_data: Tuple[Tensor, Tensor], val_data: Tuple[Tensor, Tensor]):
        self.train_data = train_data
        self.val_data = val_data
