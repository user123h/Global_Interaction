import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict

try:
    from utils import linear_interpolation_coefficient
except ImportError:
    from utils import linear_interpolation_coefficient

class FirstLinearLayer(nn.Module):

    def __init__(self, args, layer_type, sparsity_type, initial_weights=None, alpha_interpolation=None, **kwargs):
        super().__init__()

        self.args = args
        self.layer_type = layer_type
        self.sparsity_type = sparsity_type
        self.alpha_interpolation = alpha_interpolation 
        if layer_type=="diet":
            
            if self.args.wpn_embedding_type == 'zero': 
                print("Creating WPN that always returns zero")
                try:
                    from weight_predictors import ZeroWeightPredictorNetwork
                except ImportError:
                    from weight_predictors import ZeroWeightPredictorNetwork
                self.wpn = ZeroWeightPredictorNetwork(args)
            else: 
                try:
                    from weight_predictors import WeightPredictorNetwork
                except ImportError:
                    from weight_predictors import WeightPredictorNetwork
                self.wpn = WeightPredictorNetwork(args, kwargs['wpn_embedding_matrix'])
            self.weights = nn.Parameter(torch.zeros(args.feature_extractor_dims[0], args.num_features), requires_grad=True)
        elif layer_type=="diet_gnn":
            # ADVANCED WEIGHT PREDICTOR NETWORK
            print(" Creating AdvancedWeightPredictorNetwork")
            assert alpha_interpolation and 0 < alpha_interpolation <= 1
            try:
                from weight_predictors import AdvancedWeightPredictorNetwork
            except ImportError:
                from weight_predictors import AdvancedWeightPredictorNetwork
            self.wpn = AdvancedWeightPredictorNetwork(args, kwargs['wpn_embedding_matrix'])
            
            # used for interpolation with learned weights
            self.weights = nn.Parameter(torch.zeros(args.feature_extractor_dims[0], args.num_features), requires_grad=True)
        elif layer_type=="standard":
            if initial_weights is None: # initialize weights with kaiming initialization
                self.weights = nn.Parameter(torch.zeros(args.feature_extractor_dims[0], args.num_features), requires_grad=True)
                nn.init.kaiming_normal_(self.weights, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
            else:                       # initialize weights with specified initial weights
                self.weights = nn.Parameter(initial_weights, requires_grad=True)
        else:
            raise ValueError("Invalid first layer type")


        # auxiliary layer after the matrix multiplication
        self.bias_first_layer = nn.Parameter(torch.zeros(args.feature_extractor_dims[0]))
        self.layers_after_matrix_multiplication = nn.Sequential(*[
            nn.LeakyReLU(),
            nn.BatchNorm1d(args.feature_extractor_dims[0]),
            nn.Dropout(args.dropout_rate)
        ])

        # SPARSITY REGULARIZATION for the first layer
        if sparsity_type=='global':
            if args.sparsity_method=='sparsity_network':
                print("Creating Sparsity network")
                self.sparsity_model = SparsityNetwork(args, kwargs['wpn_embedding_matrix'])
            else:
                raise Exception("Sparsity method not valid")
        else:
            self.sparsity_model = None

    def forward(self, x, iteration=None):
        """
        Input:
            x: (batch_size x num_features)
        """

        # COMPUTE WEIGHTS FIRST LAYER
        if self.layer_type in ["diet", "diet_gnn", "diet_quadrann"]:
            alpha_interpolation = self.alpha_interpolation
            if self.args.winit_first_layer_interpolation_scheduler=='linear':
                alpha_interpolation *= linear_interpolation_coefficient(
                    max_iterations = self.args.winit_first_layer_interpolation_end_iteration,
                    iteration = iteration
                )

            if alpha_interpolation == 0: # no WPN, only learned weights
                W = self.weights
            else: # interpolation between WPN-based weight and learned weights
                W_wpn = self.wpn()

                if W_wpn.shape != self.weights.shape:
                    print("W_wpn.shape", W_wpn.shape)
                    print("self.weights.shape", self.weights.shape)
                    raise Exception("W_wpn.shape != self.weights.shape")
                W = alpha_interpolation * W_wpn  + (1 - alpha_interpolation) * self.weights

        elif self.layer_type=="standard":
            W = self.weights # W has size (K x D)

        # APPLY SPARSITY WEIGHTS (from WPFS paper)
        if self.args.sparsity_type==None:
            all_sparsity_weights = None

            hidden_rep = F.linear(x, W, self.bias_first_layer)
        
        elif self.args.sparsity_type=='global':
            all_sparsity_weights = self.sparsity_model(None) 	# Tensor (D, )
            # print("all_sparsity_weights", all_sparsity_weights.shape)
            # print("self.args.num_features", self.args.num_features)
            # print("all_sparsity_weights.shape", all_sparsity_weights.shape)

            assert all_sparsity_weights.shape[0]==self.args.num_features and len(all_sparsity_weights.shape)==1
            W = torch.matmul(W, torch.diag(all_sparsity_weights))

            hidden_rep = F.linear(x, W, self.bias_first_layer)
  
        RESULT = self.layers_after_matrix_multiplication(hidden_rep)

        return RESULT, all_sparsity_weights


# For WPFS baseline
class SparsityNetwork(nn.Module):
	"""
	Sparsity network

	Architecture
	- same 4 hidden layers of 100 neurons as the DietNetwork (for simplicity)
	- output layer: 1 neuron, sigmoid activation function
	- note: the gating network in LSNN used 3 hidden layers of 100 neurons

	Input
	- gene_embedding: gene embedding (batch_size, embedding_size)
	Output
	- sigmoid value (which will get multiplied by the weights associated with the gene)
	"""
	def __init__(self, args, embedding_matrix):
		super().__init__()
		
		print(f"Initializing SparsityNetwork with embedding_matrix of size {embedding_matrix.size()}")
		
		self.args = args
		self.register_buffer('embedding_matrix', embedding_matrix) # store the static embedding_matrix

		layers = []
		dim_prev = args.wpn

		for _, dim in enumerate(args.diet_network_dims):
			layers.append(nn.Linear(dim_prev, dim))
			layers.append(nn.LeakyReLU())
			layers.append(nn.BatchNorm1d(dim))
			layers.append(nn.Dropout(args.dropout_rate))

			dim_prev = dim
		
		layers.append(nn.Linear(dim, 1))
		self.network = nn.Sequential(*layers)

		if args.mixing_layer_size:
			mixing_layers = []

			layer1 = nn.Linear(args.num_features, args.mixing_layer_size, bias=False)
			nn.init.uniform_(layer1.weight, -0.005, 0.005)
			mixing_layers.append(layer1)

			mixing_layers.append(nn.LeakyReLU())

			if args.mixing_layer_dropout:
				mixing_layers.append(nn.Dropout(args.mixing_layer_dropout))
			
			layer2 = nn.Linear(args.mixing_layer_size, args.num_features, bias=False)
			nn.init.uniform_(layer2.weight, -0.005, 0.005)
			mixing_layers.append(layer2)

			self.mixing_layers = nn.Sequential(*mixing_layers)
		else:
			self.mixing_layers = None

	def forward(self, input):
		"""
		Input:
		- input: Tensor of patients (B, D)

		Returns:
		if args.sparsity_type == 'global':
			- Tensor of sigmoid values (D)
		"""
		if self.args.sparsity_type == 'global':
			out = self.network(self.embedding_matrix) # (D, 1)]

			print("SparsityNetwork: global sparsity")
			print("self.embedding_matrix", self.embedding_matrix.shape)
			print("out", out.shape)

			if self.mixing_layers:
				out = self.mixing_layers(out.T).T # input of size (1, D) to the linear layer

			out = torch.sigmoid(out)
			return torch.squeeze(out, dim=1) 		  # (D)


class ConcreteLayer(nn.Module):

    def __init__(self, args, input_dim, output_dim, is_diet_layer=False, wpn_embedding_matrix=None):
        """
        - input_dim (int): dimension of the input
        - output_dim (int): number of neurons in the layer
        """
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temp_start = 10
        self.temp_end = 0.01
        # the iteration is used in annealing the temperature
        # 	it's increased with every call to sample during training
        self.current_iteration = 0 
        self.anneal_iterations = args.concrete_anneal_iterations # maximum number of iterations for the temperature optimization

        self.is_diet_layer = is_diet_layer
        if is_diet_layer:
            # if diet layer, then initialize a weight predictor matrix
            assert wpn_embedding_matrix is not None
            try:
                from weight_predictors import WeightPredictorNetwork
            except ImportError:
                from weight_predictors import WeightPredictorNetwork
            self.wpn = WeightPredictorNetwork(args, wpn_embedding_matrix)
        else:
            self.alphas = nn.Parameter(torch.zeros(output_dim, input_dim), requires_grad=True)
            torch.nn.init.xavier_normal_(self.alphas, gain=1) # Glorot normalization, following the original CAE implementation
        
    def get_temperature(self):
        # compute temperature		
        if self.current_iteration >= self.anneal_iterations:
            return self.temp_end
        else:
            return self.temp_start * (self.temp_end / self.temp_start) ** (self.current_iteration / self.anneal_iterations)

    def sample(self):
        """
        Sample from the concrete distribution.
        """
        # Increase the iteration counter during training
        if self.training:
            self.current_iteration += 1

        temperature = self.get_temperature()

        alphas = self.wpn() if self.is_diet_layer else self.alphas # alphas is a K x D matrix

        # sample from the concrete distribution
        if self.training:
            samples = F.gumbel_softmax(alphas, tau=temperature, hard=False) # size K x D
            assert samples.shape == (self.output_dim, self.input_dim)
        else: 			# sample using argmax
            index_max_alphas = torch.argmax(alphas, dim=1) # size K
            samples = torch.zeros(self.output_dim, self.input_dim).cuda()
            samples[torch.arange(self.output_dim), index_max_alphas] = 1.

        return samples

    def forward(self, x):
        """
        - x (batch_size x input_dim)
        """
        mask = self.sample()   	# size (number_neurons x input_dim)
        x = torch.matmul(x, mask.T) 		# size (batch_size, number_neurons)
        return x, None # return additional None for compatibility


class Decoder(nn.Module):
    def __init__(self, args, wpn):
        super().__init__()
        assert wpn!=None, "The decoder is used only with a WPN (because it's only used within the DietNetwork)"

        self.wpn = wpn
        self.bias = nn.Parameter(torch.zeros(args.num_features,))

    def forward(self, hidden_rep):
        W = self.wpn().T # W has size D x K

        return F.linear(hidden_rep, W, self.bias) 


def compute_triplet_loss(embeddings, labels, margin=1.0, distance_p=2.0, swap=False, mining_strategy='hard'):
    import torch.nn.functional as F
    
    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    batch_size = embeddings.size(0)
    if distance_p == 1:
        distances = torch.cdist(embeddings, embeddings, p=1)
    elif distance_p == 2:
        distances = torch.cdist(embeddings, embeddings, p=2)
    else:
        distances = torch.cdist(embeddings, embeddings, p=distance_p)
    labels = labels.unsqueeze(1)
    same_label_mask = (labels == labels.T).float()
    diff_label_mask = (labels != labels.T).float()
    if mining_strategy == 'hard':
        triplet_loss = _hard_triplet_mining(distances, same_label_mask, diff_label_mask, margin, swap)
    elif mining_strategy == 'semi_hard':
        triplet_loss = _semi_hard_triplet_mining(distances, same_label_mask, diff_label_mask, margin)
    elif mining_strategy == 'random':
        triplet_loss = _random_triplet_mining(distances, same_label_mask, diff_label_mask, margin)
    else:
        raise ValueError(f"Unknown excavation strategy: {mining_strategy}")
    
    return triplet_loss

def _hard_triplet_mining(distances, same_label_mask, diff_label_mask, margin, swap=False):
    batch_size = distances.size(0)
    triplet_loss = torch.tensor(0.0, device=distances.device, requires_grad=True)
    valid_triplets = 0
    
    for i in range(batch_size):
        positive_mask = same_label_mask[i] * (torch.arange(batch_size, device=distances.device) != i)
        if positive_mask.sum() == 0:
            continue
        negative_mask = diff_label_mask[i]
        if negative_mask.sum() == 0:
            continue
        positive_distances = distances[i] * positive_mask + (-1e9) * (1 - positive_mask)
        hardest_positive_idx = torch.argmax(positive_distances)
        hardest_positive_dist = distances[i, hardest_positive_idx]
        negative_distances = distances[i] * negative_mask + 1e9 * (1 - negative_mask)
        hardest_negative_idx = torch.argmin(negative_distances)
        hardest_negative_dist = distances[i, hardest_negative_idx]
        basic_loss = hardest_positive_dist - hardest_negative_dist + margin
        if swap:
            swap_loss = hardest_positive_dist - distances[hardest_negative_idx, hardest_positive_idx] + margin
            loss = torch.max(basic_loss, swap_loss)
        else:
            loss = basic_loss
        
        if loss > 0:
            triplet_loss = triplet_loss + F.relu(loss)
            valid_triplets += 1
    if valid_triplets > 0:
        return triplet_loss / valid_triplets
    else:
        return torch.tensor(0.0, device=distances.device, requires_grad=True)

def _semi_hard_triplet_mining(distances, same_label_mask, diff_label_mask, margin):
    batch_size = distances.size(0)
    triplet_loss = torch.tensor(0.0, device=distances.device, requires_grad=True)
    valid_triplets = 0
    
    for i in range(batch_size):
        positive_mask = same_label_mask[i] * (torch.arange(batch_size, device=distances.device) != i)
        if positive_mask.sum() == 0:
            continue
        negative_mask = diff_label_mask[i]
        if negative_mask.sum() == 0:
            continue
        positive_indices = torch.where(positive_mask)[0]
        for pos_idx in positive_indices:
            pos_dist = distances[i, pos_idx]
            
            
            semi_hard_mask = negative_mask * (distances[i] > pos_dist) * (distances[i] < pos_dist + margin)
            
            if semi_hard_mask.sum() > 0:
                semi_hard_distances = distances[i] * semi_hard_mask + 1e9 * (1 - semi_hard_mask)
                hardest_negative_idx = torch.argmin(semi_hard_distances)
                neg_dist = distances[i, hardest_negative_idx]
                
                loss = pos_dist - neg_dist + margin
                if loss > 0:
                    triplet_loss = triplet_loss + F.relu(loss)
                    valid_triplets += 1
    
    if valid_triplets > 0:
        return triplet_loss / valid_triplets
    else:
        return torch.tensor(0.0, device=distances.device, requires_grad=True)

def _random_triplet_mining(distances, same_label_mask, diff_label_mask, margin):
    batch_size = distances.size(0)
    triplet_loss = torch.tensor(0.0, device=distances.device, requires_grad=True)
    valid_triplets = 0

    num_anchors = min(batch_size, 32)  
    anchor_indices = torch.randperm(batch_size, device=distances.device)[:num_anchors]
    
    for i in anchor_indices:
        positive_mask = same_label_mask[i] * (torch.arange(batch_size, device=distances.device) != i)
        if positive_mask.sum() == 0:
            continue
        negative_mask = diff_label_mask[i]
        if negative_mask.sum() == 0:
            continue
        positive_indices = torch.where(positive_mask)[0]
        negative_indices = torch.where(negative_mask)[0]
        
        if len(positive_indices) > 0 and len(negative_indices) > 0:
            pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))].item()
            neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))].item()
            
            pos_dist = distances[i, pos_idx]
            neg_dist = distances[i, neg_idx]
            
            loss = pos_dist - neg_dist + margin
            if loss > 0:
                triplet_loss = triplet_loss + F.relu(loss)
                valid_triplets += 1
    
    if valid_triplets > 0:
        return triplet_loss / valid_triplets
    else:
        return torch.tensor(0.0, device=distances.device, requires_grad=True)

class TripletLossModule(nn.Module):
    def __init__(self, margin=1.0, distance_p=2.0, swap=False, mining_strategy='hard'):
        super(TripletLossModule, self).__init__()
        self.margin = margin
        self.distance_p = distance_p
        self.swap = swap
        self.mining_strategy = mining_strategy
    
    def forward(self, embeddings, labels):
        return compute_triplet_loss(
            embeddings, labels, 
            margin=self.margin,
            distance_p=self.distance_p,
            swap=self.swap,
            mining_strategy=self.mining_strategy
        )

def extract_hidden_features(model, x):
    if hasattr(model, 'encoder_to_hidden'):
        x_processed, _ = model.first_layer(x, getattr(model, 'global_step', 0))
        if hasattr(model, 'second_layer') and model.second_layer:
            x_processed = model.second_layer(x_processed)
        hidden_features = model.encoder_to_hidden(x_processed)
        return hidden_features
    else:
        return x 