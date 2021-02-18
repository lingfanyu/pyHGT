Featurize nodes with no input data
==========================
Scripts in the folder are adapted from [NARS project](https://github.com/facebookresearch/NARS/tree/main/graph_embed).


Dependencies
--------------
- torch==1.5.1+cu101
- dgl-cu101==0.4.3.post2
- ogb==1.2.1
- dglke==0.1.0

Instructions
--------------

Convert heterogeneous graph to triplet format (src_node_id,
edge_type, dst_node_id). This step only uses the graph structure. The node
features are not used.
```bash
python3 convert_to_triplets.py --dataset mag
```

Before generating embedding, it's better to remove existing DGL-KE
checkpoints (if any) in this folder:
```bash
rm -rf ckpts
```

The bash script `train_graph_emb.sh` uses DGL-KE to train graph embedding. Feel free to
change any hyper-parameter in the script. 
```bash
bash train_graph_emb.sh mag
```

The generated embedding of all nodes will be stored in `ckpts` folder. We need
to split the graph embedding by node types and reorder back to original node
order:
```bash
python3 split_node_emb.py --dataset mag
mkdir ../TransE_mag
mv *.pt ../TransE_mag
```
