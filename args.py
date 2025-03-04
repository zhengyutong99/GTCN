import argparse
from utils import init_seed

init_seed(seed=777)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GTN', help='Model')
parser.add_argument('--base_dir', type=str, default='/dataset', help='Input_directory')
parser.add_argument('--dataset', type=str, default='Cellline-based_dataset', help='Dataset')
parser.add_argument('-EM', '--embedding_method', type=int, default='1', help='Embedding_method')
parser.add_argument('--load_trained_model', action='store_true', help="Load trained model parameters")
parser.add_argument('--best_fold', type=int, default=None, help="Specify the best run for prediction")
parser.add_argument('--epoch', type=int, default=5000, help='Training Epochs')
parser.add_argument('-ESP', '--early_stop_patience', type=int, default=300,
                    help="If loss didn't decrease, then stop epoch")
parser.add_argument('-ND', '--node_dim', type=int, default=64, help='hidden dimensions')
parser.add_argument('-NCH', '--num_channels', type=int, default=3, help='number of channels')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-WD', '--weight_decay', type=float, default=0.001, help='l2 reg')
parser.add_argument('-NL', '--num_layers', type=int, default=2, help='number of GT/FastGT layers')
parser.add_argument('-NGL', '--num_GCN_layers', type=int, default=3, help='number of GCN layers')
parser.add_argument('--runs', type=int, default=1, help='number of runs')
parser.add_argument('--folds', type=int, default=10, help='number of folds')
parser.add_argument('-CA', "--channel_agg", type=str, default='concat')
parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")
parser.add_argument("--non_local", action='store_true', help="use non local operations")
parser.add_argument("--non_local_weight", type=float, default=0, help="weight initialization for non local operations")
parser.add_argument("--beta", type=float, default=0, help="beta (Identity matrix)")
parser.add_argument('--K', type=int, default=1,
                    help='number of non-local negibors')
parser.add_argument("--pre_train", action='store_true', help="pre-training FastGT layers")
parser.add_argument('-NFL', '--num_FastGTN_layers', type=int, default=1,
                    help='number of FastGTN layers')
parser.add_argument('-POD', '--preprocess_output_dim', type=int, default=64,
                    help='Output dimension for the preprocess NN')
parser.add_argument('-NCL', '--num_classes', type=int, default=2,
                    help='Class number of GCN output')
parser.add_argument('-GI', '--gpu_id', type=int, default=3, help='GPU ID to use')
parser.add_argument('-NT', '--num_ntype', default=4, type=int, help='Number of node types')
parser.add_argument('--classifier', type=str, default='NN')
parser.add_argument('--save_dir', type=str, default='/output')

args = parser.parse_args()
