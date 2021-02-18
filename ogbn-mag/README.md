Evaluate HGT on ogbn-mag dataset with TransE graph embedding
============================

This fork serves as a baseline in the evaluation of [NARS](https://github.com/facebookresearch/NARS) model. 

The difference compared to the original [HGT model](https://github.com/acbull/pyHGT):  
In [ogbn-mag dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag), only paper nodes have input features. Here, we featurize node types that don't have input features (e.g. author, field, institution) with TransE graph embedding.

Dependencies
--------------
This model uses the folowing python packages:
- torch==1.5.1+cu101
- torch_geometric==1.5.0
	- torch_scatter==2.0.5
	- torch_sparse==0.6.7
	- torch_cluster==1.5.7
- ogb==1.2.1
- dill
- pandas
- tqdm
- sklearn
- gensim

Steps to run
-------------

### Generate TransE graph embedding
Please follow instructions in [graph_embed](./graph_embed) to generate TransE embeddings.

### Preprocess ogbn-mag dataset
The following command converts ogbn-mag dataset into the format that HGT model uses:
```bash
python3 preprocess_ogbn_mag.py --graph-emb TransE_mag
```

### Train model
```bash
python3 train_ogbn_mag.py --n_hid 512 --n_layer 5 --n_heads 8 --data_dir ./OGB_MAG.pk \
    --prev_norm --last_norm --use_RTE --conv_name hgt --sample_width 520 --sample_depth 6
```

### Evaluate model
```bash
python3 eval_ogbn_mag.py --n_hid 512 --n_layer 5 --n_heads 8 --data_dir ./OGB_MAG.pk \
    --prev_norm --last_norm --use_RTE --conv_name hgt --sample_width 520 --sample_depth 6
```

Results on ogbn-mag
-------------
| Model        | Testing Accuracy        | Validation Accuracy  | # Parameter     | Hardware         |
| ---------    | ----------------------- | ------------------   | --------------  | ---------------  |
| 5-layer HGT  | 0.4982&plusmn;0.0013    | 0.5124&plusmn;0.0046 | 26,877,657      | Tesla T4 (15GB)  |
