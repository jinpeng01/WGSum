
#WGSUM
Code for ACL 2021 paper: *[Word Graph Guided Summarization for Radiology Findings]*

==========

This repo contains the PyTorch code following [this code](https://github.com/nlpyang/PreSumm)

## Requirements

- Python 3 (tested on 3.7)
- PyTorch (tested on 1.5)

## Data

We give an example about the data in the `graph_construction/`

### Preparation


#### Graph Construction
We have given the example about the data format to construct the graph (each line is a radiology report).
You might need to change the data path to you own data path.
```
cd graph_construction
python graph_construction.py
``` 

After finish graph construction. need to run `sh precess_radiology.sh` to further process data. For this step, you can obtain more information from the link (https://github.com/nlpyang/PreSumm). Note that you also need to change the 322-324 row in src/prepro/data_builder.py to your own data.


## Training
change DATA_PATH to your data_path
To start training, run

```
sh WGSUM_Trans.sh
```

## Evaluation
change DATA_PATH Model_path to your data_path and model path
To start evaluation, run
```
sh WGSUM_test.sh
```
