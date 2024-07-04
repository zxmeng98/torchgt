import torch
import torch.distributed as dist
from multiprocessing.pool import ThreadPool
import time
from gt_sp.initialize import (
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_src_rank,
)


def sync_params_and_buffers(model):
    for name, param in model.state_dict().items():
        torch.distributed.broadcast(param.data,
                                    src=get_sequence_parallel_src_rank(),
                                    group=get_sequence_parallel_group())
        

class Reducer(object):

    def __init__(self):
        super(Reducer, self).__init__()
        self._group = {}
        self._pool = None
        self._handles = []
        self._stream = None

    def init(self, model):
        cnt = 0
        for i, (name, param) in enumerate(model.named_parameters()):
            cnt += 1
            self._group[name] = dist.new_group() 

        self._stream = torch.cuda.Stream(device=f'cuda:{torch.cuda.current_device()}')

    def reduce(self, param, name, data):
        # TODO communicate flatten tensor for high efficiency
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            group = self._group[name]
            data.div_(get_sequence_parallel_world_size())
            dist.all_reduce(data, op=dist.ReduceOp.SUM, group=group)
            # param.grad = data            

    def synchronize(self):
        torch.cuda.current_stream().wait_stream(self._stream)
