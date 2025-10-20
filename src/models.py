import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import scipy

from utils import get_labels_lists, detach_tensors, create_linear_layers
from layers import ConcreteLayer, compute_triplet_loss, extract_hidden_features

# ----------------------  DNN class for training  -------------------
class TrainingLightningModule(pl.LightningModule):
    """
    General class to be inherited by all implemented models (e.g., MLP, CAE, FsNet etc.)

    It implements general training and evaluation functions (e.g., computing losses, logging, training etc.)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.epoch_start_time = None
        self.first_epoch_logged = False

    def compute_loss(self, y_true, y_hat, x, x_hat, sparsity_weights, hidden_features=None):
        losses = {}
        losses['cross_entropy'] = F.cross_entropy(input=y_hat, target=y_true, weight=torch.tensor(self.args.class_weights, device=self.device))
        losses['reconstruction'] = F.mse_loss(x_hat, x, reduction='mean') if self.decoder else torch.zeros(1, device=self.device)

        if sparsity_weights is None:
            losses['sparsity'] = torch.tensor(0., device=self.device)
        else:
            if self.args.sparsity_regularizer=='L1':
                losses['sparsity'] = self.args.sparsity_regularizer_hyperparam * torch.norm(sparsity_weights, 1)
            elif self.args.sparsity_regularizer=='hoyer':
                hoyer_reg = torch.norm(sparsity_weights, 1) / torch.norm(sparsity_weights, 2)
                losses['sparsity'] = self.args.sparsity_regularizer_hyperparam * hoyer_reg
            else:
                raise Exception("Sparsity regularizer not valid")

        if getattr(self.args, 'use_triplet_loss', False) and hidden_features is not None:
            triplet_margin = getattr(self.args, 'triplet_margin', 1.0)
            triplet_distance_p = getattr(self.args, 'triplet_distance_p', 2.0)
            triplet_swap = getattr(self.args, 'triplet_swap', False)
            triplet_mining_strategy = getattr(self.args, 'triplet_mining_strategy', 'hard')
            
            losses['triplet'] = compute_triplet_loss(
                hidden_features, y_true,
                margin=triplet_margin,
                distance_p=triplet_distance_p,
                swap=triplet_swap,
                mining_strategy=triplet_mining_strategy
            )
        else:
            losses['triplet'] = torch.tensor(0., device=self.device)
        triplet_weight = getattr(self.args, 'triplet_weight', 0.0)
        losses['total'] = (losses['cross_entropy'] + 
                          self.args.gamma * losses['reconstruction'] + 
                          losses['sparsity'] + 
                          triplet_weight * losses['triplet'])
        
        return losses

    def log_losses(self, losses, key, dataloader_name=""):
        self.log(f"{key}/total_loss{dataloader_name}", losses['total'].item())
        self.log(f"{key}/reconstruction_loss{dataloader_name}", losses['reconstruction'].item())
        self.log(f"{key}/cross_entropy_loss{dataloader_name}", losses['cross_entropy'].item())
        self.log(f"{key}/sparsity_loss{dataloader_name}", losses['sparsity'].item())
        if 'triplet' in losses:
            self.log(f"{key}/triplet_loss{dataloader_name}", losses['triplet'].item())
        if 'graph_reconstruction' in losses:
            self.log(f'{key}/graph_reconstruction', losses['graph_reconstruction'].item())


    def log_epoch_metrics(self, outputs, key, dataloader_name=""):
        y_true, y_pred = get_labels_lists(outputs)
        from sklearn.metrics import balanced_accuracy_score
        self.log(f'{key}/balanced_accuracy{dataloader_name}', balanced_accuracy_score(y_true, y_pred))

    def on_train_epoch_start(self):
        import time
        self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batches coming from multiple dataloaders
        """
        # tabular data
        x, y_true = batch['tabular']
        y_hat, x_hat, sparsity_weights = self.forward(x)

        hidden_features = None
        if getattr(self.args, 'use_triplet_loss', False):
            hidden_features = extract_hidden_features(self, x)

        losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights, hidden_features)

        self.log_losses(losses, key='train')
        
        if isinstance(self.first_layer, ConcreteLayer):
            self.log("train/concrete_temperature", self.first_layer.get_temperature())

        # Store outputs for epoch end processing
        output = {
            'loss': losses['total'],
            'losses': detach_tensors(losses),
            'y_true': y_true,
            'y_pred': torch.argmax(y_hat, dim=1)
        }
        
        # Initialize training_step_outputs if it doesn't exist
        if not hasattr(self, 'training_step_outputs'):
            self.training_step_outputs = []
        self.training_step_outputs.append(output)
        
        return output

    def on_train_epoch_end(self):
        # Get outputs from training steps
        outputs = self.training_step_outputs if hasattr(self, 'training_step_outputs') else []
        if outputs: 
            self.log_epoch_metrics(outputs, 'train')
            
            # Calculate and log training loss for OFI callback
            train_total_loss = np.mean([output['losses']['total'].item() for output in outputs])
            self.log('train/total_loss', train_total_loss)
            
            if not self.first_epoch_logged and self.epoch_start_time is not None:
                import time
                epoch_time = time.time() - self.epoch_start_time
                print(f"第一个epoch用时: {epoch_time:.2f}秒")
                self.first_epoch_logged = True
                time.sleep(2)  
            
            self.training_step_outputs.clear()  # Clear for next epoch

    def validation_step(self, batch, batch_idx):
        x, y_true = batch['tabular']
        y_hat, x_hat, sparsity_weights = self.forward(x)
        hidden_features = None
        if getattr(self.args, 'use_triplet_loss', False):
            hidden_features = extract_hidden_features(self, x)

        losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights, hidden_features)

        # Store outputs for epoch end processing
        output = {
            'losses': detach_tensors(losses),
            'y_true': y_true,
            'y_pred': torch.argmax(y_hat, dim=1)
        }
        
        # Initialize validation_step_outputs if it doesn't exist
        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
        self.validation_step_outputs.append(output)
        
        return output

    def on_validation_epoch_end(self):
        # Get outputs from validation steps
        outputs = self.validation_step_outputs if hasattr(self, 'validation_step_outputs') else []
        if not outputs:
            return
            
        losses = {
            'total': np.mean([output['losses']['total'].item() for output in outputs]),
            'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs]),
            'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
            'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs])
        }

        if 'triplet' in outputs[0]['losses']:
            losses['triplet'] = np.mean([output['losses']['triplet'].item() for output in outputs])

        # sometimes the graph reconstruction loss is not computed
        if 'graph_reconstruction' in outputs[0]['losses']:
            losses['graph_reconstruction'] = np.mean([output['losses']['graph_reconstruction'].item() for output in outputs]),

        self.log_losses(losses, key='val')
        self.log_epoch_metrics(outputs, key='val')
        
        # Ensure val/total_loss is logged for OFI callback
        self.log('val/total_loss', losses['total'])
        
        self.validation_step_outputs.clear()  # Clear for next epoch

    def test_step(self, batch, batch_idx):
        x, y_true = batch['tabular']
        y_hat, x_hat, sparsity_weights = self.forward(x)
        
        hidden_features = None
        if getattr(self.args, 'use_triplet_loss', False):
            hidden_features = extract_hidden_features(self, x)
            
        losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights, hidden_features)

        output = {
            'losses': detach_tensors(losses),
            'y_true': y_true,
            'y_pred': torch.argmax(y_hat, dim=1),
            'y_hat': y_hat.detach().cpu().numpy()
        }
        
        # Initialize test_step_outputs if it doesn't exist
        if not hasattr(self, 'test_step_outputs'):
            self.test_step_outputs = []
        self.test_step_outputs.append(output)
        
        return output

    def on_test_epoch_end(self):
        # Get outputs from test steps
        outputs = self.test_step_outputs if hasattr(self, 'test_step_outputs') else []
        
        if not outputs:
            return
            
        ### Save losses
        losses = {
            'total': np.mean([output['losses']['total'].item() for output in outputs]),
            'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs]),
            'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),	
            'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs])
        }

        if outputs and 'triplet' in outputs[0]['losses']:
            losses['triplet'] = np.mean([output['losses']['triplet'].item() for output in outputs])

        # sometimes the graph reconstruction loss is not computed
        if outputs and 'graph_reconstruction' in outputs[0]['losses']:
            losses['graph_reconstruction'] = np.mean([output['losses']['graph_reconstruction'].item() for output in outputs]),

        self.log_losses(losses, key=self.log_test_key)
        self.log_epoch_metrics(outputs, self.log_test_key)

        #### Save prediction probabilities
        y_hat_list = [output['y_hat'] for output in outputs]
        y_hat_all = np.concatenate(y_hat_list, axis=0)
        y_hat_all = scipy.special.softmax(y_hat_all, axis=1)
        
        # Clear outputs for next test epoch
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        params = self.parameters()

        if self.args.optimizer=='adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.args.weight_decay)
        if self.args.optimizer=='adamw':
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.args.weight_decay, betas=[0.9, 0.98])

        if self.args.lr_scheduler == 'none':
            return optimizer
        else:
            print(f"Warning: lr_scheduler '{self.args.lr_scheduler}' not implemented, using no scheduler")
            return optimizer


class DNN(TrainingLightningModule):
    def __init__(self, args, first_layer, second_layer=None, decoder=None):
        """
        DNN with a feature_extractor and a final layer (with `num_classes` logits)
        :param first_layer (nn.Module): first layer of the DNN (used mainly for WPN)
        :param second_layer (nn.Module): second layer of the DNN (enables having a WPN on the second layer)
        :param nn.Module decoder: decoder (for reconstruction loss)
                If None, then don't have a reconstruction loss
        """
        super().__init__(args)

        if decoder:
            print(f'Creating {args.model} with decoder...')
        else:
            print(f'Creating {args.model} without decoder...')

        self.args = args
        self.log_test_key = None
        self.learning_rate = args.lr
        
        self.first_layer = first_layer
        self.second_layer = second_layer

        # split the layers into two parts to 
        encoder_first_layers, encoder_second_layers = create_linear_layers(
            args, args.feature_extractor_dims, args.layers_for_hidden_representation-1) # the -1 in (args.layers_for_hidden_representation - 1) is because we don't consider the first layer

        self.encoder_to_hidden = nn.Sequential(*encoder_first_layers)
        self.hidden_to_logits = nn.Sequential(*encoder_second_layers)

        self.classification_layer = nn.Linear(args.feature_extractor_dims[-1], args.num_classes)
        self.decoder = decoder

    def forward(self, x):
        x, sparsity_weights = self.first_layer(x, self.global_step)			   # pass through the first layer
        if self.second_layer:								   # pass through the second layer
            x = self.second_layer(x) 

        x = self.encoder_to_hidden(x)					       # obtain the hidden representation (defined as a layer in the network)
        x_hat = self.decoder(x) if self.decoder else None      # reconstruction
        x = self.hidden_to_logits(x)

        y_hat = self.classification_layer(x)           		   # classification, returns logits

        return y_hat, x_hat, sparsity_weights