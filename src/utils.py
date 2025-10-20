import time
import math
import torch
from torch import nn
from torch.nn.init import _calculate_correct_fan, calculate_gain
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import pytorch_lightning as pl

def get_labels_lists(outputs):
    all_y_true, all_y_pred = [], []
    for output in outputs:
        all_y_true.extend(output['y_true'].detach().cpu().numpy().tolist())
        all_y_pred.extend(output['y_pred'].detach().cpu().numpy().tolist())

    return all_y_true, all_y_pred

def detach_tensors(tensors):

    if type(tensors)==list:
        detached_tensors = list()
        for tensor in tensors:
            detached_tensors.append(tensor.detach())
        return detached_tensors
    elif type(tensors)==dict:
        detached_tensors = dict()
        for key, tensor in tensors.items():
            detached_tensors[key] = tensor.detach()
        return detached_tensors
    else:
        raise Exception("tensors must be a list or a dict")

def linear_interpolation_coefficient(max_iterations, iteration):
    if max_iterations==None or max_iterations<=0:
        raise Exception("max_iterations must be a positive integer")

    return iteration / max_iterations

def create_linear_layers(args, layer_sizes, layers_for_hidden_representation):
    """
    Args
    - layer_sizes: list of the sizes of the sizes of the linear layers
    - layers_for_hidden_representation: number of layers of the first part of the encoder (used to output the input for the decoder)

    Returns
    Two lists of Pytorch Modules (e.g., Linear, BatchNorm1d, Dropout)
    - encoder_first_part
    - encoder_second_part
    """
    encoder_first_part = []
    encoder_second_part = []
    for i, (dim_prev, dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if i < layers_for_hidden_representation:					# first part of the encoder
            encoder_first_part.append(nn.Linear(dim_prev, dim))
            encoder_first_part.append(nn.LeakyReLU())
            if args.batchnorm:
                encoder_first_part.append(nn.BatchNorm1d(dim))
            encoder_first_part.append(nn.Dropout(args.dropout_rate))
        else:														# second part of the encoder
            encoder_second_part.append(nn.Linear(dim_prev, dim))
            encoder_second_part.append(nn.LeakyReLU())
            if args.batchnorm:
                encoder_second_part.append(nn.BatchNorm1d(dim))
            encoder_second_part.append(nn.Dropout(args.dropout_rate))
        
    return encoder_first_part, encoder_second_part

def reshape_batch(batch):

    x, y = batch
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1)

    return x, y

def compute_all_metrics(args, y_true, y_pred):
    metrics = {}
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    return metrics

def compute_distance_matrix(X, metric='euclidean'):

    if metric == 'euclidean':
        X_norm = (X**2).sum(dim=1, keepdim=True)
        distance_matrix = X_norm + X_norm.t() - 2 * torch.mm(X, X.t())
        distance_matrix = torch.sqrt(torch.clamp(distance_matrix, min=0))
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")
    
    return distance_matrix 

def tile_weights(args, embeddings):

    if args.feature_extractor_dims[0] > embeddings.shape[0]:
        embeddings = embeddings.repeat(args.feature_extractor_dims[0]//embeddings.shape[0], 1)
        embeddings = torch.cat([embeddings, embeddings[:args.feature_extractor_dims[0] % embeddings.shape[0], :]], dim=0)
   
    return embeddings

def compute_kaiming_normal_std(weights, a=0.01, mode='fan_out', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(weights, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return std

def compute_nmf_embeddings(X, embeddings_size, ratio_diff_to_neighbor=0.05):

    print("Calculating NMF-based weight initialization...")
    from sklearn.decomposition import NMF
    
    X_positive = X - X.min() + 1e-8
    
    nmf = NMF(n_components=embeddings_size, random_state=42, max_iter=100)
    try:
        W = nmf.fit_transform(X_positive.cpu().numpy())
        H = nmf.components_
        
        embeddings = torch.tensor(H, dtype=torch.float32)
        
    except Exception as e:
        embeddings = torch.randn(embeddings_size, X.shape[1])
    embedding_info = {'method': 'nmf_based', 'components': embeddings_size}
    
    return embeddings, embedding_info

# Weight Predictor Network Factory
class WeightPredictorNetworkFactory:
    
    @staticmethod
    def create(args, embedding_matrix, variant=None):
        if variant is None:
            variant = getattr(args, 'wpn_variant', 'improved')
            
        if variant == 'improved':
            return ImprovedWeightPredictorNetwork(
                args, 
                embedding_matrix,
                dropout_rate=getattr(args, 'wpn_dropout_rate', 0.4),
                noise_scale=getattr(args, 'wpn_noise_scale', 0.1),
                use_spectral_norm=getattr(args, 'wpn_enable_spectral_norm', True),
                enable_batch_norm=getattr(args, 'wpn_enable_batch_norm', True)
            )
        elif variant == 'lightweight':
            return LightweightWeightPredictorNetwork(args, embedding_matrix)
        elif variant == 'adaptive':
            return AdaptiveWeightPredictorNetwork(args, embedding_matrix)
        elif variant == 'advanced':
            return AdvancedWeightPredictorNetwork(args, embedding_matrix)
        else:
            raise ValueError(f"Unknown variant: {variant}")

def create_model(args, data_module=None):

    pl.seed_everything(args.seed_model_init_and_training, workers=True)
    
    wpn_embedding_matrix = data_module.get_embedding_matrix(args.wpn_embedding_type, args.wpn_embedding_size)

    if args.gamma > 0:
        wpn_decoder = WeightPredictorNetworkFactory.create(args, wpn_embedding_matrix)
        from models import Decoder
        decoder = Decoder(args, wpn_decoder)
    else:
        decoder = None
    
    std_kaiming_first_layer = compute_kaiming_normal_std(torch.zeros(args.feature_extractor_dims[0], args.num_features))

    if args.model=='fsnet':
        from models import ConcreteLayer, DNN
        concrete_layer = ConcreteLayer(args, args.num_features, args.feature_extractor_dims[0], is_diet_layer=True, wpn_embedding_matrix=wpn_embedding_matrix)

        return DNN(args, concrete_layer, decoder=decoder)

    elif args.model=='cae': # Supervised Autoencoder
        from models import ConcreteLayer, DNN
        concrete_layer = ConcreteLayer(args, args.num_features, args.feature_extractor_dims[0])

        return DNN(args, concrete_layer)

    elif args.model in ['dnn', 'dietdnn']:
        initial_weights = None
        
        if args.model=='dnn':
            layer_type = 'standard'
        elif args.model=='dietdnn':
            layer_type = 'diet'
        
        if args.winit_initialisation=='awpn': 
            embedding_size = getattr(args, 'wpn_embedding_size', 50)  
            num_features = args.num_features
            
            wpn_embedding_matrix = torch.randn(num_features, embedding_size) * 0.01
    
            from layers import FirstLinearLayer
            from models import DNN
            first_layer = FirstLinearLayer(args,
                layer_type='diet_gnn', alpha_interpolation=args.winit_first_layer_interpolation, sparsity_type=args.sparsity_type,
                wpn_embedding_matrix=wpn_embedding_matrix, std_kaiming_normal=std_kaiming_first_layer)

            return DNN(args, first_layer, decoder=decoder)

        elif args.winit_initialisation=='pca':
            print("Creating PCA-based weight initialization....")
            start_time = time.time()
            from sklearn.decomposition import PCA
            
            n_samples, n_features = data_module.X_train_raw.shape
            max_components = min(n_samples, n_features)
            desired_components = args.feature_extractor_dims[0]
            
            if desired_components > max_components:
                n_components = max_components
            else:
                n_components = desired_components
                
            pca = PCA(n_components=n_components)
            pca.fit(data_module.X_train_raw)
            initial_weights = torch.tensor(pca.components_, dtype=torch.float32)
            
            if n_components < desired_components:
                target_weights = torch.zeros(desired_components, n_features)
                target_weights[:n_components] = initial_weights
                if n_components < desired_components:
                    remaining_rows = desired_components - n_components
                    target_weights[n_components:] = torch.randn(remaining_rows, n_features) * 0.01
                initial_weights = target_weights
            
            initial_weights -= initial_weights.mean()
            initial_weights *= std_kaiming_first_layer / initial_weights.std().item()

        elif args.winit_initialisation=='nmf':
            start_time = time.time()
            initial_weights, _ = compute_nmf_embeddings(data_module.X_train_raw, embeddings_size=args.feature_extractor_dims[0])
            initial_weights = tile_weights(args, initial_weights)
            initial_weights -= initial_weights.mean()
            initial_weights *= std_kaiming_first_layer / initial_weights.std().item()


        elif args.winit_initialisation=='wl': 
            start_time = time.time()
            initial_weights, _ = compute_nmf_embeddings(data_module.X_train_raw, embeddings_size=args.feature_extractor_dims[0])
            initial_weights = tile_weights(args, initial_weights)
            initial_weights -= initial_weights.mean()
            initial_weights *= std_kaiming_first_layer / initial_weights.std().item()

        elif args.winit_initialisation=='kaiming': 

            initial_weights = torch.empty(args.feature_extractor_dims[0], args.num_features)
            torch.nn.init.kaiming_normal_(initial_weights, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
            
            from layers import FirstLinearLayer
            from models import DNN
            first_layer = FirstLinearLayer(args, layer_type=layer_type, sparsity_type=args.sparsity_type,
                            wpn_embedding_matrix=wpn_embedding_matrix,
                            initial_weights=initial_weights, alpha_interpolation=args.winit_first_layer_interpolation)
            return DNN(args, first_layer, decoder=decoder)
            
        elif args.winit_initialisation=='xavier':  
            initial_weights = torch.empty(args.feature_extractor_dims[0], args.num_features)
            torch.nn.init.xavier_normal_(initial_weights, gain=1.0)
            
            from layers import FirstLinearLayer
            from models import DNN
            first_layer = FirstLinearLayer(args, layer_type=layer_type, sparsity_type=args.sparsity_type,
                            wpn_embedding_matrix=wpn_embedding_matrix,
                            initial_weights=initial_weights, alpha_interpolation=args.winit_first_layer_interpolation)
            return DNN(args, first_layer, decoder=decoder)

        from layers import FirstLinearLayer
        from models import DNN
        first_layer = FirstLinearLayer(args, layer_type=layer_type, sparsity_type=args.sparsity_type,
                        wpn_embedding_matrix=wpn_embedding_matrix,
                        initial_weights=initial_weights, alpha_interpolation=args.winit_first_layer_interpolation)

        return DNN(args, first_layer, decoder=decoder)
    else:
        raise Exception(f"The model ${args.model}$ is not supported")

class ImprovedWeightPredictorNetwork(nn.Module):
    def __init__(self, args, embedding_matrix, dropout_rate=0.4, noise_scale=0.1, use_spectral_norm=True, enable_batch_norm=True):
        super().__init__()
        self.args = args
        # Simple placeholder implementation
        self.register_buffer('embedding_matrix', embedding_matrix)
        self.weight_generator = nn.Linear(embedding_matrix.size(1), args.feature_extractor_dims[0])
        
    def forward(self):
        W = self.weight_generator(self.embedding_matrix)
        return W.T

class LightweightWeightPredictorNetwork(nn.Module):
    def __init__(self, args, embedding_matrix):
        super().__init__()
        self.args = args
        self.register_buffer('embedding_matrix', embedding_matrix)
        self.weight_generator = nn.Linear(embedding_matrix.size(1), args.feature_extractor_dims[0])
        
    def forward(self):
        W = self.weight_generator(self.embedding_matrix)
        return W.T

class AdaptiveWeightPredictorNetwork(nn.Module):
    def __init__(self, args, embedding_matrix):
        super().__init__()
        self.args = args
        self.register_buffer('embedding_matrix', embedding_matrix)
        self.weight_generator = nn.Linear(embedding_matrix.size(1), args.feature_extractor_dims[0])
        
    def forward(self):
        W = self.weight_generator(self.embedding_matrix)
        return W.T

class AdvancedWeightPredictorNetwork(nn.Module):
    def __init__(self, args, embedding_matrix):
        super().__init__()
        self.args = args
        self.register_buffer('embedding_matrix', embedding_matrix)
        self.weight_generator = nn.Linear(embedding_matrix.size(1), args.feature_extractor_dims[0])
        
    def forward(self):
        W = self.weight_generator(self.embedding_matrix)
        return W.T 