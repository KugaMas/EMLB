import time
from abc import ABC, abstractmethod
from utils.event_utils import *
import utils.cdn_utils as cdn


class EventDenoisors(ABC):
    def __init__(self, size, use_polarity=True, excl_hotpixel=True):
        super().__init__()
        self.name           = 'Template'
        self.annotation     = 'Template'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.size  = size
        self.model = None

    @staticmethod
    def pre_prosess(self, ev, size):
        if self.use_polarity is False:
            ev = event_polarity_removal(ev, size)
        if self.excl_hotpixel is False:
            ev = event_hotpixel_removal(ev, size)

        ts, x, y, p = np.split(ev, [1, 2, 3], axis=1)
        ts = ts.astype(np.int64)
        x = x.astype(np.uint16)
        y = y.astype(np.uint16)
        p = p.astype(np.bool_)

        return ts, x, y, p
    
    @abstractmethod
    def run(self, ev, fr):
        pass


class dwf(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 num_thr=1, dis_thr=10, double_mode=True, m_len=8):
        self.name           = 'DWF'
        self.annotation     = 'Double Window Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel
        
        self.mLen = m_len
        self.numThr = num_thr
        self.disThr = dis_thr
        self.duoMode = double_mode

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model  = cdn.dwf(size[0], size[1], self.numThr, self.disThr, self.duoMode, self.mLen)
        return model.run(ts, x, y, p)

class mlpf(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True):
        self.name           = 'MLPF'
        self.annotation     = 'Multi Layer Perceptron Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model  = cdn.mlpf(size[0], size[1])
        return model.run(ts, x, y, p)


def Denoisor(idx, args):
    model = eval(args.denoisors[idx].lower())
    return model(args.use_polarity, args.excl_hotpixel, *args.params[idx])
