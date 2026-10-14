"""独立早停状态；不改历史固定轮数训练和已锁定源码。"""
from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    patience: int = 10
    min_epochs: int = 15
    min_delta: float = 1e-5
    best: float = math.inf
    best_epoch: int = -1
    anchor: float = math.inf
    stale: int = 0
    last_epoch: int = -1

    def observe(self, epoch: int, value: float) -> tuple[bool, bool]:
        if self.patience<1 or self.min_epochs<0 or self.min_delta<0 or not math.isfinite(value) or epoch!=self.last_epoch+1:
            raise ValueError('invalid/nonsequential early stopping observation')
        self.last_epoch=epoch
        # 保存严格最小值；早停计数按有实际幅度的改善，二者分别核验。
        save=value<self.best
        if save:self.best,self.best_epoch=value,epoch
        if value<self.anchor-self.min_delta:
            self.anchor,self.stale=value,0
        else:self.stale+=1
        return save,epoch>=self.min_epochs and self.stale>=self.patience
