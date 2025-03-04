# GTCN-DCA

This repository is for **"Encoding Drug-Target-Pathway-Disease Profiles for Drug-Cancer Association Prediction using Graph Transformer-Convolution Networks"**, including the source code and datasets.

## Dataset

The dataset consists of **nodes, edges, node features, and labels**. It can be downloaded from **[Google Drive](https://drive.google.com/drive/folders/1nhiLSCuJfLu2f9QartgXbkWnSluukbGV?usp=sharing)**.

Our proposed **cell line-based drug repositioning dataset** includes the following node types:
- **2,614** drugs (D)  
- **20,501** targets (T)  
- **8,208** pathways (P)  
- **455** diseases (I)  

The node lists are stored in `/data/Node`, with the following files and naming conventions:
- `Drug_list.csv` → **DrugBank IDs**  
- `Target_list.csv` → **Gene Names**  
- `Pathway_list.csv` → **Disease Name + KEGG IDs**  
- `Disease_list.csv` → **Disease Name + GDSC IDs**  

Additionally, the dataset contains **edges, node features, and labels**, which are stored in `/data/Cellline-based_dataset`:
- **1,222,904** drug-drug interactions  
- **21,970** drug-target interactions  
- **7,589,721** drug-pathway associations  
- **196,105** target-target interactions  
- **18,909,648** target-pathway associations  
- **35,635** drug-disease associations  

Both **edges and labels** require preprocessing before they can be used as model input. The following scripts should be used for preprocessing:
📌 **Edge preprocessing script:** `Edge_preprocess.ipynb`  
📌 **Label preprocessing script:** `Label_preprocess.ipynb`

---

## Requirements

To set up the environment, run the following commands:

```bash
conda create -n GTCN-DCA python=3.11.8
conda activate GTCN-DCA
pip install -r requirements.txt
```

---

## Training

To train and evaluate the model, use the following command:

```bash
python main.py --model GTN --dataset Cellline-based_dataset --gpu_id {DEVICE ID} --save_dir {SAVED PATH}
```

### Optional Arguments:
- `--embedding_method` (`-EM`) → **Embedding method (default: 1), 0 stands for random, 1 stands for customized biological meaning encoding**
- `--load_trained_model` → **Load trained model parameters**
- `--best_fold` → **Specify the best run for prediction**
- `--epoch` (`-E`) → **Number of training epochs (default: 5000)**
- `--early_stop_patience` (`-ESP`) → **Early stopping patience (default: 300)**
- `--node_dim` (`-ND`) → **GCN output feature dimension (default: 64)**
- `--num_channels` (`-NCH`) → **Number of heads (default: 3)**
- `--num_layers` (`-NL`) → **Number of GT layers (default: 2)**
- `--num_GCN_layers` (`-NGL`) → **Number of GCN layers (default: 3)**
- `--lr` → **Learning rate (default: 0.001)**
- `--weight_decay` (`-WD`) → **L2 regularization (default: 0.001)**
- `--folds` → **Number of k-fold cross-validation folds (default: 10)**
- `--channel_agg` (`-CA`) → **Aggregation method for channels (default: 'concat')**
- `--remove_self_loops` → **Remove self-loops in the graph**
- `--num_GTN_layers` (`-NFL`) → **Number of GTN layers (default: 1)**
- `--preprocess_output_dim` (`-POD`) → **Output dimension for the preprocess NN (default: 64)**
- `--num_classes` (`-NCL`) → **Number of classes in the GCN output (default: 2)**
- `--num_ntype` (`-NT`) → **Number of node types (default: 4)**
- `--classifier` → **Classifier type (default: 'NN')**

For example, to train the model with 10-fold cross-validation, **5000 epochs**, **learning rate of 0.001**, and **64 hidden dimensions**, run:

```bash
python main.py --model GTN --dataset Cellline-based_dataset --gpu_id 0 --folds 10 --epoch 5000 --lr 0.001 --node_dim 64 --save_dir ./results/
```
