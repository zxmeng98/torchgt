# Sequence Parallelism for Graph Transformer 
A pytorch implementation of Graph Transformers with sequence parallelism.

Model implementation is based on "Do Transformers Really Perform Bad for Graph Representation" (NeurIPS'21) [[paper]](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html) [[github]](https://github.com/microsoft/Graphormer). 

Node-level attn bias is from "Gophormer: Ego-Graph Transformer for Node Classification" [[arxiv]](https://arxiv.org/abs/2110.13094)

## Run the code  
Environment: base

### Graph-level
main_graph_level.py: train graph-level task (e.g. ZINC) in sequence parallel way. **(NOTE!! evaluation复用forward过程在GPU上做)**

```bash
torchrun --nproc_per_node=4 main_sp_graph_level.py
```
 
### Node-level
Use preprocess_data.py for data downloading and preprocessing

main_sp.py: train in sequence parallel way. Scripts: 

```bash
torchrun --nproc_per_node=4 main_sp.py --dataset aminer --seq_len 1024 
torchrun --nproc_per_node=4 main_sp_sparse.py --dataset aminer --seq_len 1024 # with sparse attn choice
```

main_ori.py: train the graph transformer in a batched dataloder way.

main_rnd_1seq.py: train on long sequences, 1 sequence = 1 batch of random nodes.

main_rnd_1seq_evalall.py: 
- train on long sequences, 1 seq = 1 batch of random nodes, batch_size = seq length (eval on full-graph)
- run reddit/amazon2m cpu OOM on calculating full_attn_bias

## Notes

现在node-level任务去掉bias产生的话，每个iter预处理时间主要是geb_sub_edge_index，大概1s （30M边）

Graph-level任务evaluation要复用sp的model forward，收敛速度和收敛的值才正确！！

CPU evaluation very slow when seq len is large.

pubmed/physics小数据集：attn_bias_dim = 6

reddit/amazon 100k以上大数据集：attn_bias_dim = 1


## TODO

- graph/n0de-level所有的get_Batch放在训练总前面，不要每个iter都要计算

- global token no merge之后注释的tensor形状都要改, graph/node-level get_batch都要对应修改adjust_edge_index

- Bias 优化为edge_index存储形式

- 大图上面 Malnet-tiny数据集，sp情况下，collator非常耗时，wrapper也耗时（加上计算attn bias需要的一些数据之后）。后续怎么优化？

- sp+sparse：拆分后[9,9,8,8]这种情况edge_index的对齐。最简单的处理方式，collate里面保证只有最后一个sub_seq需要pad [9,9,9,8]。

## BUG

- 异步hook gradients all-reduce集群上有问题 


