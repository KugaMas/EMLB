from abc import ABC, abstractmethod
from .utils.event_utils import *


class objective_denoisors(ABC):
    def __init__(self, use_polarity=True, excl_hotpixel=True):
        super().__init__()
        self.name           = 'Template'
        self.annotation     = 'Template'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

    @staticmethod
    def pre_prosess(self, ev, size):
        if self.excl_hotpixel:
            ev = event_hotpixel_removal(ev, size)
        if ~self.use_polarity:
            ev = event_polarity_removal(ev, size)
        return ev
    
    @abstractmethod
    def run(self, ev, fr, size):
        pass


class mlpf(objective_denoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True):
        self.name           = 'MLPF'
        self.annotation     = 'Multilayer Perceptron denoising Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

    def run(self, ev, fr, size):
        ev = self.pre_prosess(ev, size)
        return ev


def Denoisor(denoisor, params):
    model = eval(denoisor.lower())
    return model(*params)
