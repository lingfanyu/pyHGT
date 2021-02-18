Featurize Nodes With No Input
==========================
Scripts in the folder are adapted from [NARS project](https://github.com/facebookresearch/NARS).


Dependencies
--------------
- torch==1.5.1+cu101
- dgl-cu101==0.4.3.post2
- ogb==1.2.1
- dglke==0.1.0

Instructions
--------------

The following commands generate TransE embedding for ogbn-mag dataset using DGL-KE:
```bash
dataset=mag
```

Convert heterogeneous graph to triplet format (src_node_id,
edge_type, dst_node_id). This step only uses the graph structure. The node
features are not used.
```bash
python3 convert_to_triplets.py --dataset ${dataset}
```

Before generating embedding, it's better to remove existing DGL-KE
checkpoints (if any) in this folder:
```bash
rm -rf ckpts
```

The shell script `train_graph_emb.sh` uses DGL-KE to train graph embedding. The
default configuration is what we used to evaluate our paper. But feel free to
change any setting in the script like the graph embedding model, training
hyper-parameters. The training takes about 40 mins to finish on a Tesla V100 GPU.
You can also speed it up by allowing DGL-KE to use more GPUs.
```bash
bash train_graph_emb.sh ${dataset}
```

The generated embedding of all nodes will be stored in `ckpts` folder. We need
to split the graph embedding by node types and reorder back to original node
order:
```bash
python3 split_node_emb.py --dataset ${dataset}
mkdir ../TransE_${dataset}
mv *.pt ../TransE_${dataset}
```
