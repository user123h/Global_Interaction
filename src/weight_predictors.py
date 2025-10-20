import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, use_lightweight=True):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        output = self.layer_norm(attn_output + x)
        
        if squeeze_output:
            output = output.squeeze(1)
        return output

class CrossModalAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(key_dim, query_dim)
        self.layer_norm = nn.LayerNorm(query_dim)

    def forward(self, query, key, value, mask=None):
        if query.dim() == 2:
            query = query.unsqueeze(1) 
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            squeeze_out = True
        else:
            squeeze_out = False

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        attn_output, _ = self.attention(q, k, v, key_padding_mask=mask)
        output = self.layer_norm(attn_output + query) 

        if squeeze_out:
            output = output.squeeze(1)
        return output

class FeatureAttention(nn.Module):
    def __init__(self, num_features, hidden_dim=64, use_softmax=True, residual=True):
        super().__init__()
        self.use_softmax = use_softmax
        self.residual = residual

        self.attention_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, x):
        scores = self.attention_net(x) 
        if self.use_softmax:
            weights = F.softmax(scores, dim=-1)
        else:
            weights = torch.sigmoid(scores)

        if self.residual:
            output = x + x * weights  
        else:
            output = x * weights

        return output, weights

class KNNClusteringModule(nn.Module):
    def __init__(self, num_features: int, n_clusters: int = 8, n_neighbors: int = 5, output_dim: int = 32):
        super().__init__()
        self.num_features = num_features
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.output_dim = output_dim
        
        self.cluster_centers = nn.Parameter(
            torch.randn(n_clusters, num_features) * 0.1,
            requires_grad=True
        )
        
        self.cluster_encoder = nn.Sequential(
            nn.Linear(n_clusters + n_neighbors + 3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.cluster_weights = nn.Parameter(torch.ones(n_clusters), requires_grad=True)
        
    def compute_soft_clustering(self, x: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(x, self.cluster_centers)
        cluster_assignments = F.softmax(-distances / self.temperature, dim=1)
        weighted_assignments = cluster_assignments * self.cluster_weights.unsqueeze(0)
        return weighted_assignments
    
    def compute_knn_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        pairwise_distances = torch.cdist(x, x)
        
        noise = torch.randn_like(pairwise_distances) * 1e-6
        pairwise_distances_noisy = pairwise_distances + noise
        
        pairwise_distances_noisy.fill_diagonal_(float('inf'))
        
        k = min(self.n_neighbors, batch_size - 1)
        if k <= 0:
            return torch.zeros(batch_size, self.n_neighbors, device=x.device)
        
        _, knn_indices = torch.topk(pairwise_distances_noisy, k, dim=1, largest=False)
        knn_distances = torch.gather(pairwise_distances, 1, knn_indices)
        
        if k < self.n_neighbors:
            padding = torch.zeros(batch_size, self.n_neighbors - k, device=x.device)
            knn_distances = torch.cat([knn_distances, padding], dim=1)
        
        return knn_distances
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cluster_assignments = self.compute_soft_clustering(x)
        knn_distances = self.compute_knn_features(x)
        
        local_mean = torch.mean(x, dim=1, keepdim=True).expand(-1, 1)
        local_std = torch.std(x, dim=1, keepdim=True).expand(-1, 1) + 1e-8
        local_entropy = -torch.sum(F.softmax(x, dim=1) * F.log_softmax(x, dim=1), dim=1, keepdim=True)
        local_stats = torch.cat([local_mean, local_std, local_entropy], dim=1)
        
        clustering_features = torch.cat([
            cluster_assignments, knn_distances, local_stats
        ], dim=1)
        
        encoded_features = self.cluster_encoder(clustering_features)
        
        intra_cluster_distance = self._compute_intra_cluster_distance(x, cluster_assignments)
        inter_cluster_distance = self._compute_inter_cluster_distance()
        clustering_loss = intra_cluster_distance - 0.1 * inter_cluster_distance
        
        return {
            'clustering_features': encoded_features,
            'cluster_assignments': cluster_assignments,
            'knn_distances': knn_distances,
            'local_statistics': local_stats,
            'clustering_loss': clustering_loss,
            'intra_cluster_distance': intra_cluster_distance,
            'inter_cluster_distance': inter_cluster_distance
        }
    
    def _compute_intra_cluster_distance(self, x: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
        total_distance = 0.0
        for i in range(self.n_clusters):
            center = self.cluster_centers[i]
            distances = torch.norm(x - center.unsqueeze(0), dim=1)
            weighted_distances = distances * assignments[:, i]
            total_distance += weighted_distances.mean()
        return total_distance / self.n_clusters
    
    def _compute_inter_cluster_distance(self) -> torch.Tensor:
        center_distances = torch.cdist(self.cluster_centers, self.cluster_centers)
        mask = ~torch.eye(self.n_clusters, dtype=torch.bool, device=center_distances.device)
        inter_distances = center_distances[mask]
        return inter_distances.mean()

class AdvancedStatisticalExtractor(nn.Module):
    def __init__(self, num_features: int, enable_causal: bool = True, output_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.enable_causal = enable_causal
        self.output_dim = output_dim
        
        self.stats_dim = 8
        self.stat_weights = nn.Parameter(torch.ones(8))
        
        self.stats_projection = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def compute_advanced_statistics(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_features = x.shape
        device = x.device
        
        mean = x.mean()
        std = x.std() + 1e-8
        var = x.var() + 1e-8
        
        flattened = x.flatten()
        sorted_vals, _ = flattened.sort()
        n = flattened.size(0)
        if n % 2 == 1:
            median = sorted_vals[n // 2]
        else:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        
        centered = x - mean
        skew_approx = (centered ** 3).mean() / (std ** 3 + 1e-8)
        kurt_approx = (centered ** 4).mean() / (std ** 4 + 1e-8) - 3
        
        global_range = flattened.max() - flattened.min()
        q25_idx = n // 4
        q75_idx = 3 * n // 4
        iqr = sorted_vals[q75_idx] - sorted_vals[q25_idx] if n > 4 else torch.tensor(0.0, device=device)
        
        stats = torch.stack([
            mean, median, std, var, 
            skew_approx, kurt_approx, global_range, iqr
        ])
        
        return stats.unsqueeze(0).expand(batch_size, -1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.compute_advanced_statistics(x)
        weighted_stats = stats * self.stat_weights.unsqueeze(0)
        output = self.stats_projection(weighted_stats)
        return output

class MultiLevelFeatureExtractor(nn.Module):
    def __init__(self, num_features: int, mulitilevel_dim: int = 64, num_levels: int = 3):
        super().__init__()
        self.num_features = num_features
        self.mulitilevel_dim = mulitilevel_dim
        self.num_levels = num_levels
        
        self.level_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, mulitilevel_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mulitilevel_dim, mulitilevel_dim)
            )
            for _ in range(num_levels)
        ])
        
        self.cross_level_attention = CrossModalAttention(mulitilevel_dim, mulitilevel_dim, mulitilevel_dim // 2)
        
        self.fusion = nn.Sequential(
            nn.Linear(mulitilevel_dim * num_levels, mulitilevel_dim),
            nn.ReLU(),
            nn.Linear(mulitilevel_dim, mulitilevel_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        level_features = []
        
        for level_extractor in self.level_extractors:
            level_feat = level_extractor(x)
            level_features.append(level_feat)
        
        attended_features = []
        for i, feat in enumerate(level_features):
            for j, other_feat in enumerate(level_features):
                if i != j:
                    attended = self.cross_level_attention(feat, other_feat, other_feat)
                    attended_features.append(attended)
                else:
                    attended_features.append(feat)
        
        combined = torch.cat(level_features, dim=-1)
        mulitilevel_features = self.fusion(combined)
        
        return mulitilevel_features


class AdvancedWeightPredictorNetwork(nn.Module):
    
    def __init__(
        self, 
        args,
        embedding_matrix,
        enable_mulitilevel_learning: bool = True,
        enable_contrastive: bool = False,
        enable_causal: bool = True,
        enable_knn_clustering: bool = True,
        hidden_dim: int = 128,
        contrastive_temperature: float = 0.1
    ):
        super().__init__()
        print(f"Initializing AdvancedWeightPredictorNetwork with embedding_matrix of size {embedding_matrix.size()}")
        
        self.args = args
        self.register_buffer('embedding_matrix', embedding_matrix)
        
        self.num_features = embedding_matrix.size(0)  
        self.embedding_size = embedding_matrix.size(1) 
        self.target_output_size = args.diet_network_dims[-1]  
        
        self.enable_mulitilevel_learning = enable_mulitilevel_learning
        self.enable_contrastive = enable_contrastive
        self.enable_causal = enable_causal
        self.enable_knn_clustering = enable_knn_clustering
        self.hidden_dim = hidden_dim
        
        self.stats_extractor = AdvancedStatisticalExtractor(
            num_features=self.embedding_size,
            enable_causal=enable_causal,
            output_dim=64
        )
        
        if enable_knn_clustering:
            self.knn_clustering = KNNClusteringModule(
                num_features=self.embedding_size,
                n_clusters=8,
                n_neighbors=5,
                output_dim=32
            )
            clustering_dim = 32
        else:
            self.knn_clustering = None
            clustering_dim = 0
        
        if enable_mulitilevel_learning:
            self.mulitilevel_learner = MultiLevelFeatureExtractor(
                num_features=self.embedding_size,
                mulitilevel_dim=64,
                num_levels=2
            )
            mulitilevel_dim = 64
        else:
            self.mulitilevel_learner = None
            mulitilevel_dim = 0
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        attention_input_dim = hidden_dim // 2 + 64 + clustering_dim + mulitilevel_dim
        self.feature_attention = FeatureAttention(
            num_features=attention_input_dim,
            hidden_dim=32
        )

        self.weight_generator = nn.Sequential(
            nn.Linear(attention_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.target_output_size) 
        )
        
        if enable_contrastive:
            self.contrastive_projector = nn.Sequential(
                nn.Linear(attention_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
            self.contrastive_temperature = contrastive_temperature
        else:
            self.contrastive_projector = None
        
        if enable_causal:
            self.causal_module = nn.Sequential(
                nn.Linear(attention_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
        else:
            self.causal_module = None
        

        self._initialize_weights()
    
    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self) -> torch.Tensor:

        embeddings = self.embedding_matrix 
        batch_size = embeddings.shape[0]  
        device = embeddings.device
        
        stats_features = self.stats_extractor(embeddings) 

        encoded_features = self.feature_encoder(embeddings) 

        clustering_loss = 0.0
        if self.knn_clustering is not None:
            clustering_result = self.knn_clustering(embeddings)
            clustering_features = clustering_result['clustering_features'] 
            clustering_loss = clustering_result['clustering_loss']
        else:
            clustering_features = torch.zeros(batch_size, 0, device=device)
        
        if self.mulitilevel_learner is not None:
            mulitilevel_features = self.mulitilevel_learner(embeddings)  
        else:
            mulitilevel_features = torch.zeros(batch_size, 0, device=device)

        combined_features = torch.cat([
            encoded_features,  
            stats_features,      
            clustering_features, 
            mulitilevel_features      
        ], dim=-1)
        
        attended_features, attention_weights = self.feature_attention(combined_features)

        W = self.weight_generator(attended_features)  
        
        if self.args.softmax_diet_network:
            W = F.softmax(W, dim=1)  
        
        return W.T
    
    def get_auxiliary_losses(self):
        if hasattr(self, '_last_clustering_loss'):
            return {'clustering_loss': self._last_clustering_loss}
        return {}


WeightPredictorNetwork = AdvancedWeightPredictorNetwork


class ZeroWeightPredictorNetwork(nn.Module):
    """
    Outputs a weight matrix W with all zeros
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return torch.zeros(self.args.feature_extractor_dims[0], self.args.num_features, device=device) 