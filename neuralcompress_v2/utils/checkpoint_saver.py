"""
"""

from pathlib import Path

import torch
from torch.jit import trace, save

class CheckpointSaver:
    def __init__(self,
                 checkpoint_path,
                 frequency = -1,
                 benchmark = float('inf'),
                 prefix    = 'mod',
                 jit       = True):

        self.checkpoint_path = Path(checkpoint_path)

        self.frequency = frequency
        self.benchmark = benchmark
        self.prefix    = prefix
        self.jit       = jit

    def __call__(self, model, *, epoch, metric, data = None):

        if self.jit:
            traced_model = trace(model, data)

        # save last
        name = self.checkpoint_path/f'{self.prefix}_last.pth'
        if self.jit:
            save(traced_model, name)
        else:
            torch.save(model.state_dict(), name)

        # scheduled save
        if self.frequency > 0:
            if epoch % self.frequency == 0:
                name = self.checkpoint_path/f'{self.prefix}_{epoch}.pth'
                if self.jit:
                    save(traced_model, name)
                else:
                    torch.save(model.state_dict(), name)

        # save best model
        if metric < self.benchmark:
            self.benchmark = metric
            name = self.checkpoint_path/f'{self.prefix}_best.pth'
            if self.jit:
                save(traced_model, name)
            else:
                torch.save(model.state_dict(), name)

