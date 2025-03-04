import torch
import numpy as np
import torch.nn as nn
import pickle
import argparse
import copy
import os
import pandas as pd
import time
import tensorflow as tf
import warnings

from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score as sk_f1_score, roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from torch_geometric.utils import add_self_loops
from torch.utils.tensorboard import SummaryWriter

from args import args
from model_gtn import GTNs
from utils import init_seed, _norm, f1_score, get_features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '0'：顯示所有訊息, '1'：隱藏INFO訊息, '2'：隱藏INFO和WARNING訊息, '3'：僅顯示ERROR訊息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 讀取 label.pkl 並使用 StratifiedKFold 分割
def load_and_split_labels(args):

    labels = 'labels.pkl'
    # labels = 'D-I_labels_GDSC_IC50_mean.pkl'
    # 讀取 label.pkl 檔案 ##############################
    with open(f'{args.base_dir}/%s/{labels}' % args.dataset, 'rb') as f:
        labels = pickle.load(f)
    
    # 將 labels 的 Array 0 和 Array 1 合併
    combined_array = np.vstack(labels)
    X = combined_array[:, :2]  # 特徵為前兩列
    y = combined_array[:, 2]   # 標籤為最後一列

    # 使用 StratifiedKFold 分割，n_splits 直接用 args.folds
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_data = []  # 保存每個 fold 的分割結果
    
    for train_idx, test_idx in skf.split(X, y):
        train_data = combined_array[train_idx]
        test_data = combined_array[test_idx]
        fold_data.append((train_data, test_data))
    
    # 保存每個 fold 的內容
    save_folds(args, fold_data)
    
    return fold_data

def save_folds(args, fold_data):
    save_dir = os.path.join(args.save_dir, "folds")
    os.makedirs(save_dir, exist_ok=True)

    for i, (train_data, test_data) in enumerate(fold_data):
        fold_dir = os.path.join(save_dir, f"fold_{i + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # 保存 training 資料
        train_df = pd.DataFrame(train_data, columns=["Drug Index", "Cancer Index", "Label"])
        train_path = os.path.join(fold_dir, "training.csv")
        train_df.to_csv(train_path, index=False)
        
        # 保存 testing 資料
        test_df = pd.DataFrame(test_data, columns=["Drug Index", "Cancer Index", "Label"])
        test_path = os.path.join(fold_dir, "testing.csv")
        test_df.to_csv(test_path, index=False)

        print(f"Fold {i + 1} saved: training -> {train_path}, testing -> {test_path}")

def mask_test_links_by_labels(edges, test_labels):
    """
    根據測試集標籤將 D-I 和 I-D 矩陣中指定位置的值設為 0。
    
    Args:
        edges: List，包含所有的鄰接矩陣（SparseMatrix 格式）。
        test_labels: Numpy Array，測試集的標籤，形狀為 [n, 3]，每行是 (drug_index, cell_line_index, label)。
    
    Returns:
        edges: 修改後的鄰接矩陣列表。
    """
    from scipy.sparse import dok_matrix

    # 提取 D-I 和 I-D 矩陣
    # DI_matrix = edges[3]  # D-I 矩陣
    # ID_matrix = edges[12]  # I-D 矩陣
    
    # 預測D-I ##############################
    DI_matrix = edges[3]  # D-I 矩陣
    ID_matrix = edges[9]  # I-D 矩陣
    
    # REDDA
    # DI_matrix = edges[4]
    # ID_matrix = edges[20]

    # 預測D-T
    # DI_matrix = edges[1]  # D-T 矩陣
    # ID_matrix = edges[4]  # T-D 矩陣

    # 將矩陣轉換為可修改的 DOK 格式
    DI_matrix = DI_matrix.todok()
    ID_matrix = ID_matrix.todok()

    # 遍歷測試集的標籤，刪除對應的邊
    for drug_index, cell_line_index, _ in test_labels:
        # D-I 矩陣：藥物 → 細胞株
        if (drug_index, cell_line_index) in DI_matrix:
            DI_matrix[drug_index, cell_line_index] = 0

        # I-D 矩陣：細胞株 → 藥物
        if (cell_line_index, drug_index) in ID_matrix:
            ID_matrix[cell_line_index, drug_index] = 0

    # 返回修改後的矩陣，並轉回 CSR 格式
    # edges[3] = DI_matrix.tocsr()
    # edges[12] = ID_matrix.tocsr()

    # 預測D-I ##############################
    edges[3] = DI_matrix.tocsr()
    edges[9] = ID_matrix.tocsr()
    
    # REDDA
    # edges[3] = DI_matrix.tocsr()
    # edges[9] = ID_matrix.tocsr()

    # 預測D-T
    # edges[1] = DI_matrix.tocsr()
    # edges[4] = ID_matrix.tocsr()

    return edges

def create_adjacent_matrix(edges):
    # 為每種邊類型建立鄰接矩陣
    A = []
    for i, edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor).to(device)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor).to(device)
        # normalize each adjacency matrix
        if args.model == 'GTN' and args.dataset != 'AIRPORT':
            edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr=value_tmp, fill_value=1e-20, num_nodes=num_nodes)
            deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp, value_tmp))
    edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(torch.cuda.LongTensor).to(device)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor).to(device)
    A.append((edge_tmp, value_tmp))
    
    return A

# 定義計算評估指標的函數 20240903_增加recall, specificity, precision
def calculate_metrics(y_true, y_pred, num_classes, average='macro'):
    y_true_np = y_true.detach().cpu().numpy()  # 將真實標籤轉換為 numpy 陣列
    y_pred_np = y_pred.detach().cpu().numpy()  # 將預測結果轉換為 numpy 陣列
    y_pred_class = np.argmax(y_pred_np, axis=1)  # 獲取預測類別

    # 如果類別數大於2，則對輸出進行二值化處理以進行多類別 AUC 計算
    if num_classes > 2:
        y_true_bin = label_binarize(y_true_np, classes=list(range(num_classes)))
        y_pred_prob = y_pred_np
        auc = roc_auc_score(y_true_bin, y_pred_prob, multi_class='ovr')
        aupr = average_precision_score(y_true_bin, y_pred_prob, average=average)
    else:
        y_true_bin = y_true_np
        y_pred_prob = y_pred_np[:, 1]  # 對於二分類問題，使用正類別的概率
        auc = roc_auc_score(y_true_bin, y_pred_prob)
        aupr = average_precision_score(y_true_bin, y_pred_prob)

    f1 = sk_f1_score(y_true_np, y_pred_class, average=average)
    accuracy = accuracy_score(y_true_np, y_pred_class)
    recall = recall_score(y_true_np, y_pred_class, average=average)
    precision = precision_score(y_true_np, y_pred_class, average=average)

    # 計算specificity
    cm = confusion_matrix(y_true_np, y_pred_class)
    tn = cm[0, 0]  # true negatives
    fp = cm[0, 1]  # false positives
    specificity = tn / (tn + fp)
    
    # 計算 MCC
    mcc = matthews_corrcoef(y_true_np, y_pred_class)

    return f1, auc, aupr, accuracy, recall, specificity, precision, mcc

# 生成參數字符串(不重要)
def get_parameter_string(args):
    exclude = ['base_dir','dataset', 'best_fold', 'remove_self_loops', 'non_local', 'non_local_weight', 'beta', 'K', 'pre_train', 'load_trained_model', 'gpu_id', 'save_dir']  # 排除不需要的參數
    params = [f"{k[0:3]}{str(v)[:5]}" for k, v in vars(args).items() if k not in exclude]  # 生成參數列表
    return "_".join(params)

# 保存預測結果
def save_predictions(args, model, A, node_features, drug_indices, cancer_indices, fold, save_dir, type_mask, num_classes, batch_size=1000, epoch=None):
# def save_predictions(args, model, A, node_features, drug_indices, cancer_indices, fold, save_dir, num_classes, batch_size=1000):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S") # 獲取當前時間戳
    
    drug_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/202401DrugBank_approved_druglist_no_lonely_withUT.csv'
    cancer_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/CellLine.txt'
    
    # 20240902_DT_prediction
    # drug_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/REDDA_drugs.csv'
    # cancer_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/REDDA_targets.csv'
    
    # 20240903_DrDi_prediction
    # cancer_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/REDDA_diseases.csv'
    
    # 20241023 202401DrugBank_sm_approved_haveMol
    # drug_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/202401DrugBank_sm_approved_haveMol_druglist.csv'
    # cancer_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/202401DrugBank_sm_approved_haveMol_cellline.csv'
    # cancer_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/all_gene_TCGA.txt'
    
    # 20241106 Bdataset
    # drug_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/Bdataset_drugs.csv'
    # cancer_list_path = '/data/zhengyutong/PharmaCoNets_GTN2/PharmaCoNets_GTN_dataset/node_info/Bdataset_disease.csv'
    
    drug_list = pd.read_csv(drug_list_path, header=None).squeeze().tolist()
    cancer_list = pd.read_csv(cancer_list_path, header=None).squeeze().tolist()

    print(f"Drug list length: {len(drug_list)}, Cancer list length: {len(cancer_list)}")

    model.eval()  # 設置模型為評估模式
    results = np.zeros((len(drug_list), len(cancer_list) * num_classes), dtype=np.float32)
    start_time = time.time() # 開始計時預測總時長

    all_pairs = [(d.item(), c.item()) for d in drug_indices for c in cancer_indices]  # 生成所有可能的 drug-cell line pair的index
    
    with torch.no_grad():
        for i in range(0, len(all_pairs), batch_size):
            batch_start_time = time.time()
            batch_pairs = all_pairs[i:i+batch_size]
            batch_drug_indices = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long).to(model.device)
            batch_cancer_indices = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long).to(model.device)

            # loss, y_train, W, updated_A, X_auto = model(A, node_features, train_drug_indices, train_cancer_indices, train_target, type_mask = type_mask, epoch=i)
            y_pair, attention_score, New_adjs, _ = model(A, node_features, batch_drug_indices, batch_cancer_indices, target=None, type_mask = type_mask)  # 進行預測
            # y_pair, attention_score, New_adjs, _ = model(A, node_features, batch_drug_indices, batch_cancer_indices, target=None)

            y_pred_prob = torch.softmax(y_pair, dim=1).cpu().numpy()  # 提取預測機率
            
            for j, (drug_index, cancer_index) in enumerate(batch_pairs): # 將drug-cancer pair的index分批送進模型預測
                drug_idx = drug_indices.tolist().index(drug_index)
                cancer_idx = cancer_indices.tolist().index(cancer_index)
                results[drug_idx, cancer_idx*num_classes:(cancer_idx+1)*num_classes] = y_pred_prob[j] # 紀錄不同種分類數量(--num_class)下drug-cell line pair機率

                if cancer_idx < len(cancer_list):
                    results[drug_idx, cancer_idx*num_classes:(cancer_idx+1)*num_classes] = y_pred_prob[j]
                else:
                    print(f"Invalid cancer_idx: {cancer_idx}, which is out of range.")

            batch_end_time = time.time() # 結束計時預測總時長
            batch_time = batch_end_time - batch_start_time
            print(f"Processed {i + len(batch_pairs)} out of {len(all_pairs)} pairs. Batch time: {batch_time:.2f} seconds")
    
    # 紀錄預測時的attention score
    count = 0
    for w in attention_score:
        if isinstance(w, list):
            count += 1
            print(f"Channel-{count} attention score : ", end="")
            for idx, w_sub in enumerate(w):
                if idx < len(w) - 1:
                    print(w_sub, end='\t')
                else:
                    print(w_sub, end='')  # 最後一個元素後不加 tab
            print()  # 換行
    
    # for i in range(args.num_channels):
    #     print(f"Channel-{i} attention score : {attention_score[i]}") # 紀錄預測時的attention score

    end_time = time.time()
    total_time = end_time - start_time

    formatted_results = pd.DataFrame(results, index=drug_list, columns=[
        f"{cancer}_{cls}" for cancer in cancer_list for cls in [f"Class{i}" for i in range(num_classes)]
    ]).round(3)

    if epoch == None:
        save_path = os.path.join(save_dir, f'{current_time}_predictions_fold_{fold}.csv')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_path = os.path.join(save_dir,'training_process_predictions', f'{current_time}_predictions_fold_{fold}_epoch_{epoch}.csv')
        os.makedirs(save_dir + '/training_process_predictions/', exist_ok=True)
    
    formatted_results.to_csv(save_path)

    print(f"Predictions saved to {save_path}")
    print(f"Total prediction time: {(total_time/60):.2f} minutes")
    print(f"Average time per pair: {total_time/len(all_pairs):.4f} seconds")
    
    return New_adjs

# 保存鄰接矩陣
def save_adj_matrix(args, A, fold):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 獲取當前時間戳
    
    adj_matrices = {}
    for i, (edge_index, edge_attr) in enumerate(A):
        adj_matrices[f'edge_type_{i}_indices'] = edge_index.cpu().numpy()
        adj_matrices[f'edge_type_{i}_values'] = edge_attr.cpu().numpy()
    
    save_path = os.path.join(save_dir, f'{current_time}_adj_matrix_fold_{fold}.npz') # 儲存學完的鄰接矩陣
    os.makedirs(save_dir, exist_ok=True)
    np.savez(save_path, **adj_matrices)

# 訓練模型
def training(args, A, num_edge_type, node_features, in_dims, type_mask, save_dir):
    # 載入並分割 labels
    fold_data = load_and_split_labels(args)
    
    # 初始化最終分數列表
    final_f1, final_micro_f1, final_auc, final_aupr, final_accuracy, final_recall, final_specificity, final_precision, final_MCC = [], [], [], [], [], [], [], [], []
    
    tmp = None
    best_f1_score = 0
    best_loss = 1
    best_fold = None
    
    for run in range(args.runs + (1 if args.pre_train else 0)):  # 增加一個額外的運行以進行預訓練
        # for fold in range(args.folds):
        for fold, (train_data, test_data) in enumerate(fold_data):
            if fold != 2:
                continue
            args.fold = fold
            print(f"Training Fold {args.fold}/{args.folds-1}")
            
            # Mask test edge ################
            # # 獲取測試集節點索引
            # test_drug_indices = test_data[:, 0].astype(int)  # 測試集的藥物索引
            # test_cellline_indices = test_data[:, 1].astype(int)  # 測試集的細胞株索引

            # 創建新的鄰接矩陣，移除測試集的連結
            A = mask_test_links_by_labels(copy.deepcopy(edges), test_data)
            A = create_adjacent_matrix(A)
            # 使用修改後的鄰接矩陣進行訓練
            # loss, y_train, W, updated_A, X_auto = model(modified_edges, node_features, train_drug_indices, train_cancer_indices, target=train_target, type_mask=type_mask, epoch=i)
            #################################
        
            
            train_drug_indices = torch.from_numpy(np.array(train_data)[:, 0]).type(torch.cuda.LongTensor).to(device)
            train_cancer_indices = torch.from_numpy(np.array(train_data)[:, 1]).type(torch.cuda.LongTensor).to(device)
            train_target = torch.from_numpy(np.array(train_data)[:, 2]).type(torch.cuda.LongTensor).to(device)

            test_drug_indices = torch.from_numpy(np.array(test_data)[:, 0]).type(torch.cuda.LongTensor).to(device)
            test_cancer_indices = torch.from_numpy(np.array(test_data)[:, 1]).type(torch.cuda.LongTensor).to(device)
            test_target = torch.from_numpy(np.array(test_data)[:, 2]).type(torch.cuda.LongTensor).to(device)
            
            if args.embedding_method == 0:
                num_nodes = node_features.shape[0]
                preprocess_input_dim = node_features.shape[1]
                feats_dim_list = None
            elif args.embedding_method == 1:
                num_nodes = type_mask.shape[0]
                preprocess_input_dim = None
                feats_dim_list = in_dims

            # 初始化模型
            if args.model == 'GTN': # 使用加速版的GTN
                if args.pre_train and run == 1: # 在正式訓練前預先訓練
                    pre_trained GTNs = []
                    for layer in range(args.num_GTN_layers):
                        pre_trained GTNs.append(copy.deepcopy(model GTNs[layer].layers))
                
                while len(A) > num_edge_type:
                    del A[-1]  # 刪除額外的邊
                
                # 定義模型變數，賦予各種層數大小以及最後預測的種類等等
                model = GTNs(args = args,
                                num_edge_type=len(A),
                                w_in = args.preprocess_output_dim,
                                num_class = args.num_classes,
                                num_nodes = num_nodes,
                                preprocess_input_dim = preprocess_input_dim,
                                preprocess_output_dim=args.preprocess_output_dim,
                                feats_dim_list = in_dims,
                                ).to(device)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 設置優化器 (這裡使用常用的Adam)

            # 計算並列印模型參數數量
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'Total number of trainable parameters: {num_params}')

            model.to(device) # 把模型送進GPU裡
            
            if args.dataset == 'PPI':
                loss = nn.BCELoss().cuda()
            else:
                loss = nn.CrossEntropyLoss().cuda().to(device) # 設定loss函數種類
            
            Ws = [] # 初始化attention score矩陣
            
            # 初始化各種performance參數
            best_test_loss = float('inf')
            best_train_loss = float('inf')
            best_train_f1, best_micro_train_f1, best_train_auc, best_train_aupr, best_train_accuracy, best_train_recall, best_train_specificity, best_train_precision, best_train_MCC = 0, 0, 0, 0, 0, 0, 0, 0, 0
            best_test_f1, best_micro_test_f1, best_test_auc, best_test_aupr, best_test_accuracy, best_test_recall, best_test_specificity, best_test_precision, best_test_MCC = 0, 0, 0, 0, 0, 0, 0, 0, 0
            
            # 初始化 Tensorboard 記錄器(用來監看訓練過程)
            log_dir = os.path.join(save_dir, 'tensorboard_logs', f'Run_{run}_Fold_{fold}')  # 訓練狀態記錄目錄
            writer = SummaryWriter(log_dir=log_dir)
            
            no_improvement_counter = 0 # 初始化提前停止的計數器 (參數為--early_stop_patience)
            
            # 進行訓練
            for i in range(args.epoch):
                model.zero_grad()
                model.train()
                if args.model == 'GTN':
                    # 獲取模型資料
                    # 1.loss : 模型訓練損失
                    # 2.y_train : 模型答案（分類）
                    # 3.W : attention score list，總共會有--num_channels條attention score，其中最後一個數值為identity matrix的權重
                    # 4.updated_A : GTN學到的adjacent matrix
                    # 5.X_auto : 模型自動學習的node features
                    loss, y_train, W, updated_A, X_auto = model(A, node_features, train_drug_indices, train_cancer_indices, target = train_target, type_mask = type_mask, epoch=i)
                    # loss, y_train, W, updated_A, X_auto = model(A, node_features, train_drug_indices, train_cancer_indices, train_target, epoch=i)
                else:
                    loss, y_train, W = model(A, node_features, train_drug_indices, train_cancer_indices, target = train_target, type_mask = type_mask)
                    
                if args.dataset == 'PPI':
                    y_train = (y_train > 0).detach().float().cpu()
                    train_f1 = 0.0
                    sk_train_f1 = sk_f1_score(train_target.detach().cpu().numpy(), y_train.numpy(), average='micro')
                else:
                    # 整理訓練的分數
                    train_f1, train_auc, train_aupr, train_accuracy, train_recall, train_specificity, train_precision, train_mcc = calculate_metrics(train_target, y_train, num_classes=args.num_classes)
                    sk_train_f1 = sk_f1_score(train_target.detach().cpu().numpy(), np.argmax(y_train.detach().cpu().numpy(), axis=1), average='micro')
                
                # 記錄訓練損失和各種分數
                writer.add_scalar('Loss/train', loss.item(), i)
                writer.add_scalar('F1/train', train_f1, i)
                writer.add_scalar('AUC/train', train_auc, i)
                writer.add_scalar('AUPR/train', train_aupr, i)
                writer.add_scalar('Accuracy/train', train_accuracy, i)
                writer.add_scalar('Recall/train', train_recall, i)
                writer.add_scalar('Specificity/train', train_specificity, i)
                writer.add_scalar('Precision/train', train_precision, i)
                writer.add_scalar('MCC/train', train_mcc, i)
                writer.add_scalar('Micro_F1/train', sk_train_f1, i)
                
                loss.backward() # back propagation
                optimizer.step()
                model.eval() # 模型進入testing, validation階段
                
                # 驗證模型
                with torch.no_grad():
                    if args.model == 'GTN':
                        test_loss, y_test, _, _, node_embeddings = model(A, node_features, test_drug_indices, test_cancer_indices, target = test_target, type_mask = type_mask, epoch=i)
                        # test_loss, y_test, _, _, _ = model(A, node_features, test_drug_indices, test_cancer_indices, test_target, epoch=i)
                    else:
                        # test_loss, y_test, _ = model.forward(A, node_features, test_node, test_target)
                        test_loss, y_test, _ = model.forward(A, node_features, test_drug_indices, test_cancer_indices, target = test_target, type_mask = type_mask)
                    
                    #20241225 每20 epoch儲存一次特徵畫UMAP
                    # if fold == 0 and i <= 500 and i % 20 == 0:
                    #     # 保存節點嵌入特徵
                    #     embedding_save_dir = os.path.join(args.save_dir, 'Node_embeddings')
                    #     os.makedirs(embedding_save_dir, exist_ok=True)
                    #     output_path = os.path.join(embedding_save_dir, f"node_embeddings_fold{fold}_epoch{i}.npy")
                    #     np.save(output_path, node_embeddings.cpu().numpy())
                    #     print(f"Node embeddings saved to {output_path}")
                    # elif fold == 0 and i > 500 and i % 100 == 0:
                    #     # 保存節點嵌入特徵
                    #     embedding_save_dir = os.path.join(args.save_dir, 'Node_embeddings')
                    #     os.makedirs(embedding_save_dir, exist_ok=True)
                    #     output_path = os.path.join(embedding_save_dir, f"node_embeddings_fold{fold}_epoch{i}.npy")
                    #     np.save(output_path, node_embeddings.cpu().numpy())
                    #     print(f"Node embeddings saved to {output_path}")
                    
                    if args.dataset == 'PPI':
                        test_f1 = 0.0
                        y_test = (y_test > 0).detach().float().cpu()
                        sk_test_f1 = sk_f1_score(test_target.detach().cpu().numpy(), y_test.numpy(), average='micro')
                    else:
                        # 整理測試的分數
                        test_f1, test_auc, test_aupr, test_accuracy, test_recall, test_specificity, test_precision, test_mcc = calculate_metrics(test_target, y_test, num_classes=args.num_classes)
                        sk_test_f1 = sk_f1_score(test_target.detach().cpu().numpy(), np.argmax(y_test.detach().cpu().numpy(), axis=1), average='micro')
                    
                    # 記錄測試損失和各種分數
                    writer.add_scalar('Loss/test', test_loss.item(), i)
                    writer.add_scalar('F1/test', test_f1, i)
                    writer.add_scalar('AUC/test', test_auc, i)
                    writer.add_scalar('AUPR/test', test_aupr, i)
                    writer.add_scalar('Accuracy/test', test_accuracy, i)
                    writer.add_scalar('Recall/test', test_recall, i)
                    writer.add_scalar('Specificity/test', test_specificity, i)
                    writer.add_scalar('Precision/test', test_precision, i)
                    writer.add_scalar('MCC/test', test_mcc, i)
                    writer.add_scalar('Micro_F1/test', sk_test_f1, i)

                    # 保存每個epoch的attention score
                    weights_dir = os.path.join(save_dir, 'weights')
                    os.makedirs(weights_dir, exist_ok=True)
                    w_path = os.path.join(weights_dir, f'Run_{run}_Fold_{fold}_W.txt')
                    count = 0
                    with open(w_path, 'a') as w_file:
                        w_file.write(f'Epoch {i}:\n')
                        for w in W:
                            if isinstance(w, list):
                                count += 1
                                # for w_sub in w:
                                for idx, w_sub in enumerate(w):
                                    # if idx < len(w) - 1:
                                    w_cpu = w_sub.detach().cpu().numpy()
                                    np.savetxt(w_file, w_cpu, delimiter='\t', fmt='%.3f')
                            # else:
                            #     w_cpu = w.detach().cpu().numpy()
                            #     np.savetxt(w_file, w_cpu, delimiter='\t', fmt='%.3f')
                        w_file.write('\n')
                                
                # 儲存當前epoch權重（會被下一個epoch洗掉，防止存太多權重）
                model_path = os.path.join(weights_dir, f'Run_{run}_Fold_{fold}_model.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)
                
                # 檢查損失是否慢慢收斂，若當前損失低於最佳損失則提前結束的計數器重新計算，並將當前epoch視為最佳，最後儲存權重
                if test_loss.item() < best_test_loss:
                    # best_val_loss = val_loss.item()
                    best_test_loss = test_loss.item()
                    best_train_loss = loss.item()
                    best_train_f1, best_test_f1 = train_f1, test_f1
                    best_train_auc, best_test_auc = train_auc, test_auc
                    best_train_aupr, best_test_aupr = train_aupr, test_aupr
                    best_train_accuracy, best_test_accuracy = train_accuracy, test_accuracy
                    best_train_recall, best_test_recall = train_recall, test_recall
                    best_train_specificity, best_test_specificity = train_specificity, test_specificity
                    best_train_precision, best_test_precision = train_precision, test_precision
                    best_train_MCC, best_test_MCC = train_mcc, test_mcc
                    best_micro_train_f1, best_micro_test_f1 = sk_train_f1, sk_test_f1

                    best_model_path = os.path.join(save_dir, 'weights', f'Run_{run}_Fold_{fold}_best_model.pt')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, best_model_path)
                    print(f'Best model has been saved. Epoch{i}')
                    no_improvement_counter = 0
                else:
                    # 若當前損失高於最佳損失則提前結束訓練
                    no_improvement_counter += 1
                    if no_improvement_counter >= args.early_stop_patience: # 若提前停止計數器達設定值則停止訓練
                        print(f'Early stopping at epoch {i}')
                        break
                
                if i % 10 == 0:
                    writer.flush()

            # 印出當前訓練次數的performance
            print('\nRun {} Fold {}'.format(run, fold))
            print('--------------------Best Result-------------------------')
            print('Train - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}, AUC: {:.4f}, AUPR: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, Precision: {:.4f}, MCC: {:.4f}'.format(
                best_train_loss, best_train_f1, best_micro_train_f1, best_train_auc, best_train_aupr, best_train_accuracy, best_train_recall, best_train_specificity, best_train_precision, best_train_MCC))
            # print('Valid - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}, AUC: {:.4f}, AUPR: {:.4f}, Accuracy: {:.4f}'.format(
            #     best_val_loss, best_val_f1, best_micro_val_f1, best_val_auc, best_val_aupr, best_val_accuracy))
            print('Test - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}, AUC: {:.4f}, AUPR: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, Precision: {:.4f}, MCC: {:.4f}\n'.format(
                best_test_loss, best_test_f1, best_micro_test_f1, best_test_auc, best_test_aupr, best_test_accuracy, best_test_recall, best_test_specificity, best_test_precision, best_test_MCC))
            
            final_f1.append(best_test_f1)
            final_micro_f1.append(best_micro_test_f1)
            final_auc.append(best_test_auc)
            final_aupr.append(best_test_aupr)
            final_accuracy.append(best_test_accuracy)
            final_recall.append(best_test_recall)
            final_specificity.append(best_test_specificity)
            final_precision.append(best_test_precision)
            final_MCC.append(best_test_MCC)
            
            # 紀錄最佳的訓練
            # if best_test_f1 > best_f1_score:
            #     best_f1_score = best_test_f1
            #     # best_fold = run
            #     best_fold = fold
            if best_test_loss < best_loss:
                best_loss = best_test_loss
                best_fold = fold

    # 計算所有Fold的訓練performance的平均及標準差
    print('--------------------Final Result-------------------------')
    print(f'Test All performance - Macro_F1: {final_f1}\nMicro_F1: {final_micro_f1}\nAUC: {final_auc}\nAUPR: {final_aupr}\nAccuracy: {final_accuracy}\nRecall: {final_recall}\nSpecificity: {final_specificity}\nPrecision: {final_precision}\nMCC: {final_MCC}')
    print('Test - Macro_F1: {:.4f}±{:.4f}, AUC:{:.4f}±{:.4f}, AUPR:{:.4f}±{:.4f}, Micro_F1:{:.4f}±{:.4f}, Accuracy:{:.4f}±{:.4f}, Recall:{:.4f}±{:.4f}, Specificity:{:.4f}±{:.4f}, Precision:{:.4f}±{:.4f}, MCC:{:.4f}±{:.4f}'.format(
        np.mean(final_f1), np.std(final_f1), np.mean(final_auc), np.std(final_auc),
        np.mean(final_aupr), np.std(final_aupr), np.mean(final_micro_f1), np.std(final_micro_f1),
        np.mean(final_accuracy), np.std(final_accuracy), np.mean(final_recall), np.std(final_recall),
        np.mean(final_specificity), np.std(final_specificity), np.mean(final_precision), np.std(final_precision),
        np.mean(final_MCC), np.std(final_MCC)))
        
    # 關閉 TensorBoard 寫入器
    writer.close()
        
    return best_fold

# 預測結果
def prediction(args, A, num_edge_type, node_features, in_dims, type_mask, best_fold, save_dir):
    
    # parameter_str = get_parameter_string(args)
    # save_dir = os.path.join(args.save_dir, args.dataset, parameter_str)

    if args.embedding_method == 0:
        num_nodes = node_features.shape[0]
        preprocess_input_dim = node_features.shape[1]
        feats_dim_list = None
    elif args.embedding_method == 1:
        num_nodes = type_mask.shape[0]
        preprocess_input_dim = None
        feats_dim_list = in_dims
        
    # 20241115 修改，不用重複預測，直接指定特定fold即可
    # for fold in range(args.folds):
    # 初始化模型
    if args.model == 'GTN':
        while len(A) > num_edge_type:
            del A[-1] # 刪除額外的邊
        
        # 定義模型變數，賦予各種層數大小以及最後預測的種類等等
        model = GTNs(args = args,
                        num_edge_type=len(A),
                        w_in = args.preprocess_output_dim,
                        num_class = args.num_classes,
                        num_nodes = num_nodes,
                        preprocess_input_dim = preprocess_input_dim,
                        preprocess_output_dim=args.preprocess_output_dim,
                        feats_dim_list = in_dims,
                        ).to(device)
            
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # 設置優化器
    # pre_train_model_path = os.path.join(save_dir, 'weights', f'Run_{best_fold}_Fold_{fold}_best_model.pt')
    pre_train_model_path = os.path.join(save_dir, 'weights', f'Run_0_Fold_{best_fold}_best_model.pt')
    print(f"Using Fold {best_fold} to predict.")
    checkpoint = torch.load(pre_train_model_path, map_location=device)
    
    # 讀取權重準備預測
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.to(device)

    ##############################
    New_adjs = save_predictions(args, model, A, node_features,
                        # Drug indices
                        torch.arange(2614).to(device),
                        # torch.arange(894).to(device),
                        # torch.arange(269).to(device),
                        # Drug-Disease
                        # REDDA
                        # torch.arange(40646, 894 + 18877 + 20561 + 314 + 454).to(device),
                        # Bdataset
                        # torch.arange(269, 269 + 6040 + 598).to(device),

                        # Ourdataset D-I
                        torch.arange(31323, 31323 + 455).to(device),
                        # Ourdataset D-T
                        # torch.arange(2614, 2614 + 20501).to(device),

                        
                        # Drug-Target
                        # torch.arange(894, 894 + 18877).to(device),
                        best_fold,
                        save_dir,
                        type_mask,
                        num_classes=args.num_classes,  # 添加這個參數
                        batch_size=32768)
    # 儲存預測時的鄰接矩陣
    save_adj_matrix(args, New_adjs, best_fold)

# 主程序
if __name__ == '__main__':
    print(args)
    
    # Set the device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Verify available GPUs
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # if node_features.shape[1] != args.preprocess_output_dim:
        # print("Warning: preprocess_output_dim does not match the dimensionality of node_features. Adjusting node_features.")
        # node_features = np.random.randint(0, 2, size=(node_features.shape[0], args.preprocess_output_dim))
    
    edges = 'edges.pkl'
    # edges = 'initial_adjacency_matrix.pkl'
    # edges = 'DT2DI_edges_binary.pkl'
    # edges = 'DT2DI_edges_float.pkl'
    # edges = 'DT2DI_filterDistributeEqually_edges_binary.pkl'
    # edges = 'DDTanimoto_float_edges.pkl'
    with open(f'{args.base_dir}/%s/{edges}' % args.dataset,'rb') as f:
        edges = pickle.load(f) # 讀取不同種邊的adjacent matrix
        
    # if args.dataset == 'PPI':
    #     with open('/data/zhengyutong/Graph_Transformer_Networks/data/%s/ppi_tvt_nids.pkl' % args.dataset, 'rb') as fp:
    #         nids = pickle.load(fp)

    num_nodes = edges[0].shape[0]
    args.num_nodes = num_nodes
    
    A = create_adjacent_matrix(edges)
    num_edge_type = len(A)
    
    # 所有feature用一個 NN embedding
    if args.embedding_method == 0:
        with open(f'{args.base_dir}/%s/node_features.pkl' % args.dataset,'rb') as f:
            node_features = pickle.load(f) # 讀取節點特徵
        if isinstance(node_features, torch.Tensor):
            node_features = node_features.cpu().numpy()
        #########
        # node_features = np.random.randint(0, 2, size=(node_features.shape[0], args.preprocess_output_dim))
        #########
        node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor).to(device)
        type_mask, in_dims = None, None
        
    # 不同node type 用不同 NN embedding
    elif args.embedding_method == 1:
        type_mask = np.load(f'{args.base_dir}/%s/node_types.npy' % args.dataset)
        node_features, in_dims = get_features(args, type_mask)
    
    # 建立儲存結果的資料夾
    parameter_str = get_parameter_string(args)
    save_dir = os.path.join(args.save_dir, args.dataset, parameter_str)
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 訓練
    if args.load_trained_model == False:
        best_fold = training(args, A, num_edge_type, node_features, in_dims, type_mask, save_dir)
        with open(os.path.join(save_dir, 'best_fold.txt'), 'w') as f:
            f.write(str(best_fold))
    # 預測
    elif args.load_trained_model == True:
        if args.best_fold is None:
            # 沒有指定讀取哪次訓練則自動找訓練時紀錄的最佳權重
            with open(os.path.join(save_dir, 'best_fold.txt'), 'r') as f:
                best_fold = int(f.read())
        else:
            # 指定讀取哪次訓練的權重
            best_fold = args.best_fold
        prediction(args, A, num_edge_type, node_features, in_dims, type_mask, best_fold, save_dir)