# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
import numpy as np
import copy

from gt_sp.initialize import (
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_last_batch_flag,
)
from gt_sp.utils import (
    split_tensor_along_second_dim, 
    merge_global_token, 
    merge_global_token0,
    extend_global_token,
    extend_global_token0,
    copy_global_token0,
)


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        seq_world_size = get_sequence_parallel_world_size()

        input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]

        torch.distributed.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class _SeqGather(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate.
    Forward: all-gather
    Backward: split
    """ 

    @staticmethod
    def forward(ctx, input_, gather_idx):
        """Gather tensors and concatinate along the second dimension."""
        
        seq_world_size = get_sequence_parallel_world_size()
        rank = get_sequence_parallel_rank()
        ctx.gather_idx = gather_idx
        ctx.seq_world_size = seq_world_size
        ctx.rank = rank
        
        # Bypass the function if we are using only 1 GPU.
        if seq_world_size == 1:
            return input_

        # Size and dimension.
        tensor_list = [torch.empty_like(input_) for _ in range(seq_world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_, group=get_sequence_parallel_group()) # Note: can only on same size tensor

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=gather_idx).contiguous()
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Split the tensor along its second dimension and keep the
        corresponding slice."""
        # Bypass the function if we are using only 1 GPU.
        if ctx.seq_world_size == 1:
            return grad_output

        # Split along second dimension.
        input_list = split_tensor_along_second_dim(grad_output, ctx.seq_world_size)

        # Note: torch.split does not create contiguous tensors by default.
        output = input_list[ctx.rank].contiguous()

        return (output, None)


class _SeqScatter(torch.autograd.Function):
    """Split the input in head dim and keep only the corresponding chunk to the rank.
    Forward: split
    Backward: all-gather
    """ 

    @staticmethod
    def forward(ctx, input_):
        # input_: [b, n_head, s+1, s+1]
        seq_world_size = get_sequence_parallel_world_size()
        seq_parallel_world_rank = get_sequence_parallel_rank()
        
        assert input_.size()[1] % seq_world_size == 0
        dim_size = input_.size()[1] // seq_world_size
        input_list = [t.contiguous() for t in torch.split(input_, dim_size, dim=1)]
        output = input_list[seq_parallel_world_rank]
        
        return output


    @staticmethod
    def backward(ctx, grad_output):
        seq_world_size = get_sequence_parallel_world_size()
        seq_parallel_world_rank = get_sequence_parallel_rank()
        
        # Bypass the function if we are using only 1 GPU.
        if seq_world_size == 1:
            return grad_output

        # print(f'rank {seq_parallel_world_rank} {grad_output.shape}')
       
        tensor_list = [torch.empty_like(grad_output) for _ in range(seq_world_size)]
        tensor_list[seq_parallel_world_rank] = grad_output
        torch.distributed.all_gather(tensor_list, grad_output, group=get_sequence_parallel_group()) 

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=1).contiguous()
        # print(f'after rank {seq_parallel_world_rank} {output[0, :, :6, :6]}')
        # exit(0)

        return output
    

class DistributedAttention(torch.nn.Module):
    """Distributed attn with attn bias copy in each rank.
    For graph-level tasks, no global tokens

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2, # head
        gather_idx: int = 1, # s
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias: Tensor, edge_index: Tensor, attn_type, *args: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # if self.training:
        # in shape : [b, s/p, n_head, hn]
        # global token embedding (index = 0) is same for each rank
        # print(f'rank: {get_sequence_parallel_rank()}, q: {query[:, 0, :, :].view(4, 1, -1)}')
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)
        # out shape : [b, s, np, hn]

        # attn_bias, forward: split in head dim, backward: all-gather
        # in shape : [b, n_head, s/p, s+1]
        if attn_bias is not None:
            # -> [b, np, s, s]
            attn_bias_layer = _SeqAllToAll.apply(self.spg, attn_bias, 1, 2)
            # out shape : [b, np, s, s+p]
        else:
            attn_bias_layer = attn_bias
        
        # q, k, v: [b, s, np, hn]
        # -> [b, s, hp]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, attn_bias_layer, edge_index, attn_type, *args)

        # [b, s, hp] -> [b, s/p, h]
        # gather_idx: 1, scatter_idx: 2
        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

        # out e.g., [b, s/p+1, h]
        return output
    

class DistributedAttentionNodeLevel(torch.nn.Module):
    """Distributed attn with attn bias copy in each rank.
    For graph-level tasks, no global tokens

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2, # head
        gather_idx: int = 1, # s
    ) -> None:

        super(DistributedAttentionNodeLevel, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias: Tensor, edge_index: Tensor, attn_type, *args: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # if self.training:
        # in shape : [b, s/p, n_head, hn]
        # global token embedding (index = 0) is same for each rank
        # print(f'rank: {get_sequence_parallel_rank()}, q: {query[:, 0, :, :].view(4, 1, -1)}')
        if self.training:
            query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
            key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
            value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)
            # out shape : [b, s, np, hn]

            # attn_bias, forward: split in head dim, backward: all-gather
            # in shape : [b, n_head, s/p, s+1]
            if attn_bias is not None:
                # -> [b, np, s, s]
                attn_bias_layer = _SeqAllToAll.apply(self.spg, attn_bias, 1, 2)
                # out shape : [b, np, s, s+p]
            else:
                attn_bias_layer = attn_bias
        else:
            query_layer = query
            key_layer = key
            value_layer = value
            attn_bias_layer = attn_bias
        
        # q, k, v: [b, s, np, hn]
        # -> [b, s, hp]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, attn_bias_layer, edge_index, attn_type, *args)

        if self.training:
            # [b, s+p, hp] -> [b, s+p, hp]
            # context_layer = copy_global_token0(context_layer, extend_dim=1)
            
            # [b, s, hp] -> [b, s/p, h]
            # gather_idx: 1, scatter_idx: 2
            
            output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)
        else:
            output = context_layer

        # out e.g., [b, s/p+1, h]
        return output
