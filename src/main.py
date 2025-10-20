import json
import os

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

import pytorch_lightning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger

# import lightgbm as lgb  # Moved to conditional import below
from sklearn.ensemble import RandomForestClassifier

import wandb
import warnings
import sklearn
import logging
import time

from dataset import *
from models import *
from utils import create_model
from overfitting_usage_example import add_overfitting_detection_args, add_ofi_overfitting_arguments, create_overfitting_callback, create_ofi_overfitting_callback


def get_run_name(args):
	if args.model=='dnn':
		run_name = 'mlp'
	elif args.model=='dietdnn':
		run_name = 'mlp_wpn'
	else:
		run_name = args.model

	if args.sparsity_type=='global':
		run_name += '_SPN_global'

	return run_name

def create_wandb_logger(args):
	logger = WandbLogger(
		project=WANDB_PROJECT,
		group=args.group,
		job_type=args.job_type,
		tags=args.tags,
		notes=args.notes,

		log_model=args.wandb_log_model,
		settings=wandb.Settings(start_method="thread")
	)
	logger.experiment.config.update(args)	  # add configuration file

	return logger

def create_csv_logger(args):
	return CSVLogger("logs", name=f"{args.experiment_name}")

def run_experiment(args):
	args.suffix_wand_run_name = f"repeat-{args.repeat_id}__test-{args.test_split}"

	#### Load dataset
	print(f"\nInside training function")
	print(f"\nLoading data {args.dataset}...")
	data_module = create_data_module(args)
	
	print(f"Train/Valid/Test splits of sizes {args.train_size}, {args.valid_size}, {args.test_size}")
	print(f"Num of features: {args.num_features}")

	#### Intialize logging
	if args.logger == 'wandb':
		logger = create_wandb_logger(args)
		wandb.run.name = f"{get_run_name(args)}_{args.suffix_wand_run_name}_{wandb.run.id}"
	elif args.logger == 'csv':
		logger = create_csv_logger(args)
	else:
		raise ValueError(f"Unknown logger {args.logger}")


	#### Scikit-learn training
	if args.model in ['rf', 'lgb']:
		# scikit-learn expects class_weights to be a dictionary
		class_weights = {}
		for i, val in enumerate(args.class_weights):
			class_weights[i] = val
		
		if args.model == 'rf':
			model = RandomForestClassifier(n_estimators=args.rf_n_estimators, 
						min_samples_leaf=args.rf_min_samples_leaf, max_depth=args.rf_max_depth,
						class_weight=class_weights, max_features='sqrt',
						random_state=42, verbose=True)
			model.fit(data_module.X_train, data_module.y_train)

		elif args.model == 'lgb':
			params = {
				'max_depth': args.lgb_max_depth,
				'learning_rate': args.lgb_learning_rate,
				'min_data_in_leaf': args.lgb_min_data_in_leaf,

				'class_weight': class_weights,
				'n_estimators': 200,
				'objective': 'cross_entropy',
				'num_iterations': 10000,
				'device': 'gpu',
				'feature_fraction': '0.3'
					}

		import lightgbm as lgb  # Conditional import to avoid dependency issues
		model = lgb.LGBMClassifier(**params)
		model.fit(data_module.X_train, data_module.y_train,
			  eval_set=[(data_module.X_valid, data_module.y_valid)],
			   callbacks=[lgb.early_stopping(stopping_rounds=100)])

		#### Log metrics
		y_pred_train = model.predict(data_module.X_train)
		y_pred_test = model.predict(data_module.X_test)

		train_metrics = compute_all_metrics(args, data_module.y_train, y_pred_train)
		test_metrics = compute_all_metrics(args, data_module.y_test, y_pred_test)

		if args.logger == 'wandb':
			for metrics, dataset_name in zip(
				[train_metrics, test_metrics],
				["bestmodel_train", "bestmodel_test"]):
				for metric_name, metric_value in metrics.items():
					wandb.run.summary[f"{dataset_name}/{metric_name}"] = metric_value
		elif args.logger == 'csv':
			res = {}
			for metrics, dataset_name in zip(
				[train_metrics, test_metrics],
				["bestmodel_train", "bestmodel_test"]):
				for metric_name, metric_value in metrics.items():
					csv_logger.log_metrics({f"{dataset_name}/{metric_name}": metric_value})
					res[f"{dataset_name}/{metric_name}"] = [metric_value]

			pd.DataFrame(res).to_csv(f"{csv_logger.log_dir}/metrics.csv", index=False)




	#### Pytorch lightning training
	else:

		#### Set embedding size if it wasn't provided
		if args.wpn_embedding_size==-1:
			args.wpn_embedding_size = args.train_size

		args.num_tasks = args.feature_extractor_dims[-1] 			# number of output units of the feature extractor. Used for convenience when defining the GP


		if args.max_steps!=-1:
			# compute the upper rounded number of epochs to training (used for lr scheduler in DKL)
			steps_per_epoch = np.floor(args.train_size / args.batch_size)
			args.max_epochs = int(np.ceil(args.max_steps / steps_per_epoch))
			print(f"Training for max_epochs = {args.max_epochs}")


		#### Create model
		model = create_model(args, data_module)
		if args.logger == 'wandb':
			wandb.watch(model, log=args.wandb_watch, log_freq=10)
		trainer, checkpoint_callback = train_model(args, model, data_module, logger)

	
		checkpoint_path = checkpoint_callback.best_model_path
		print(f"\n\nBest model saved on path {checkpoint_path}\n\n")
		
		if args.logger == 'wandb':
			wandb.log({"bestmodel/step": checkpoint_path.split("step=")[1].split('.ckpt')[0]})

			#### Compute metrics for the best model
	model.log_test_key = 'bestmodel_train'
	trainer.test(model, dataloaders=data_module.train_dataloader(), ckpt_path=checkpoint_path)

	model.log_test_key = 'bestmodel_valid'
	trainer.test(model, dataloaders=data_module.val_dataloader(), ckpt_path=checkpoint_path)
	model.log_test_key = 'bestmodel_test'
	trainer.test(model, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path)

	if args.logger == 'wandb':
		wandb.finish()

	print("\nExiting from train function..")
	
def train_model(args, model, data_module, logger=None):
	"""
	Return 
	- Pytorch Lightening Trainer
	- checkpoint callback
	"""

	##### Train
	mode_metric = 'max' if args.metric_model_selection=='balanced_accuracy' else 'min'
	
	class DelayedModelCheckpoint(ModelCheckpoint):
		def __init__(self, min_epochs_to_save=0, **kwargs):
			super().__init__(**kwargs)
			self.min_epochs_to_save = min_epochs_to_save
		
		def on_validation_end(self, trainer, pl_module):
			if trainer.current_epoch >= self.min_epochs_to_save:
				super().on_validation_end(trainer, pl_module)
	
	checkpoint_callback = DelayedModelCheckpoint(
		monitor=f'val/{args.metric_model_selection}',
		mode=mode_metric,
		save_top_k=args.save_top_k,
		save_last=True,
		verbose=True,
		min_epochs_to_save=args.min_epochs_to_save
	)
	callbacks = [checkpoint_callback, TQDMProgressBar()]

	if args.patience_early_stopping:
		callbacks.append(EarlyStopping(
			monitor=f'val/{args.metric_model_selection}',
			mode=mode_metric,
			patience=args.patience_early_stopping,
		))
	
	# Add overfitting detection callback
	overfitting_callback = create_overfitting_callback(args)
	if overfitting_callback is not None:
		callbacks.append(overfitting_callback)
	
	# Add OFI overfitting detection callback
	ofi_overfitting_callback = create_ofi_overfitting_callback(args)
	if ofi_overfitting_callback is not None:
		callbacks.append(ofi_overfitting_callback)
	
	callbacks.append(LearningRateMonitor(logging_interval='step'))


	pl.seed_everything(args.seed_model_init_and_training, workers=True)
	trainer = pl.Trainer(
		# Training
		max_steps=args.max_steps,
		gradient_clip_val=2.5,

		# logging
		logger=logger,
		log_every_n_steps = 1,
		val_check_interval = args.val_check_interval,
		callbacks = callbacks,

		# miscellaneous
		accelerator="gpu",
		devices=1, 
		strategy="auto",  
		detect_anomaly=args.debugging,
		deterministic=args.deterministic,
	)
	# train
	trainer.fit(model, data_module)
	
	return trainer, checkpoint_callback

def parse_arguments(args=None):
	parser = argparse.ArgumentParser()

	####### Dataset
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--dataset_size', type=int, help='100, 200, 330, 400, 800, 1600')
	parser.add_argument('--dataset_feature_set', type=str, choices=['hallmark'], default='hallmark')

	####### Model
	parser.add_argument('--model', type=str, choices=['dnn', 'dietdnn','rf', 'lgb', 'fsnet', 'cae'], default='dnn')
	
	# Fix to pass a list of integers as argument to feature_extractor_dims (see https://github.com/wandb/wandb/issues/2939)
	class ParseAction(argparse.Action):
		def __call__(self, parser, namespace, values, option_string=None):
			values = list(map(int, values[0].split()))
			setattr(namespace, self.dest, values)
	parser.add_argument('--feature_extractor_dims', nargs='+', default=[100, 100, 10], action=ParseAction, # use last dimnsion of 10 following the paper "Promises and perils of DKL" 
						help='layer size for the feature extractor. If using a virtual layer,\
							  the first dimension must match it.')
	parser.add_argument('--layers_for_hidden_representation', type=int, default=2, 
						help='number of layers after which to output the hidden representation used as input to the decoder \
							  (e.g., if the layers are [100, 100, 10] and layers_for_hidden_representation=2, \
								  then the hidden representation will be the representation after the two layers [100, 100])')


	parser.add_argument('--batchnorm', type=int, default=1, help='if 1, then add batchnorm layers in the main network. If 0, then dont add batchnorm layers')
	parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate for the main network')
	parser.add_argument('--gamma', type=float, default=0, 
						help='The factor multiplied to the reconstruction error. \
							  If >0, then create a decoder with a reconstruction loss. \
							  If ==0, then dont create a decoder.')

	####### AWPN settings
	parser.add_argument('--winit_initialisation', type=str, choices=['pca', 'wl', 'nmf', 'awpn', 'quadrann'], default=None)
	parser.add_argument('--winit_first_layer_interpolation_scheduler', type=str, choices=['linear'], default=None)
	parser.add_argument('--winit_first_layer_interpolation_end_iteration', type=int, default=200)
	parser.add_argument('--winit_first_layer_interpolation', type=float, default=1.0)

	# Graph-related parameters removed - using custom weight generation


	####### BENCHMARK MODELS HYPERPARAMETERS #######

	####### Scikit-learn parameters
	parser.add_argument('--rf_n_estimators', type=int, default=500, help='number of trees in the random forest')
	parser.add_argument('--rf_max_depth', type=int, default=5, help='maximum depth of the tree')
	parser.add_argument('--rf_min_samples_leaf', type=int, default=2, help='minimum number of samples in a leaf')

	parser.add_argument('--lgb_learning_rate', type=float, default=0.1)
	parser.add_argument('--lgb_max_depth', type=int, default=1)
	parser.add_argument('--lgb_min_data_in_leaf', type=int, default=2)

	####### Sparsity for WPFS baseline
	parser.add_argument('--sparsity_type', type=str, default=None,
						choices=['global'], help="Use global or local sparsity")
	parser.add_argument('--sparsity_method', type=str, default='sparsity_network',
						choices=['sparsity_network'], help="The method to induce sparsity")
	parser.add_argument('--mixing_layer_size', type=int, help='size of the mixing layer in the sparsity network')
	parser.add_argument('--mixing_layer_dropout', type=float, help='dropout rate for the mixing layer')

	parser.add_argument('--sparsity_regularizer', type=str, default='L1',
						choices=['L1', 'hoyer'])
	parser.add_argument('--sparsity_regularizer_hyperparam', type=float, default=0,
						help='The weight of the sparsity regularizer (used to compute total_loss)')

	####### Weight predictor network (for DietNetwork, FsNet, and WPFS)
	parser.add_argument('--wpn_embedding_type', type=str, default='nmf',
						choices=['histogram', 'all_patients', 'nmf', 'svd', 'zero'],
						help='histogram = histogram x means (like FsNet)\
							  all_patients = randomly pick patients and use their gene expressions as the embedding\
							  It`s applied over data preprocessed using `embedding_preprocessing`')
	parser.add_argument('--wpn_embedding_size', type=int, default=50, help='Size of the gene embedding')

	parser.add_argument('--diet_network_dims', type=int, nargs='+', default=[100, 100, 100, 100],
						help="None if you don't want a VirtualLayer. If you want a virtual layer, \
							  then provide a list of integers for the sized of the tiny network.")
	parser.add_argument('--nonlinearity_weight_predictor', type=str, choices=['tanh', 'leakyrelu'], default='leakyrelu')
	parser.add_argument('--softmax_diet_network', type=int, default=0, dest='softmax_diet_network',
						help='If True, then perform softmax on the output of the tiny network.')
	
							
	####### Training
	parser.add_argument('--use_best_hyperparams', action='store_true', dest='use_best_hyperparams',
						help="True if you don't want to use the best hyperparams for a custom dataset")
	parser.set_defaults(use_best_hyperparams=False)

	parser.add_argument('--concrete_anneal_iterations', type=int, default=1000,
		help='number of iterations for annealing the Concrete radnom variables (in CAE and FsNet)')

	parser.add_argument('--max_steps', type=int, default=5000, help='Specify the max number of steps to train.')
	parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
	parser.add_argument('--batch_size', type=int, default=8)

	parser.add_argument('--patient_preprocessing', type=str, default='standard',
						choices=['raw', 'standard', 'minmax'],
						help='Preprocessing applied on each COLUMN of the N x D matrix, where a row contains all gene expressions of a patient.')
	parser.add_argument('--embedding_preprocessing', type=str, default='minmax',
						choices=['raw', 'standard', 'minmax'],
						help='Preprocessing applied on each ROW of the D x N matrix, where a row contains all patient expressions for one gene.')

	####### Validation
	parser.add_argument('--metric_model_selection', type=str, default='cross_entropy_loss',
						choices=['cross_entropy_loss', 'total_loss', 'balanced_accuracy'])

	parser.add_argument('--patience_early_stopping', type=int, default=200,
						help='Set number of checks (set by *val_check_interval*) to do early stopping.\
							 It will train for at least   args.val_check_interval * args.patience_early_stopping epochs')
	parser.add_argument('--val_check_interval', type=int, default=1, 
						help='number of steps at which to check the validation')


	####### Cross-validation
	parser.add_argument('--run_repeats_and_cv', action='store_true', dest='run_repeats_and_cv')
	parser.set_defaults(run_repeats_and_cv=False)
	parser.add_argument('--repeat_id', type=int, default=0, help='each repeat_id gives a different random seed for shuffling the dataset')
	parser.add_argument('--cv_folds', type=int, default=5, help="Number of CV splits")
	parser.add_argument('--test_split', type=int, default=0, help="Index of the test fold. It should be smaller than `cv_folds`")
	parser.add_argument('--valid_percentage', type=float, default=0.1, help='Percentage of training data used for validation')
							  

	####### Optimization
	parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw'], default='adamw')
	parser.add_argument('--lr_scheduler', type=str, choices=['lambda', 'none'], default='lambda')

	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--class_weight', type=str, choices=['standard', 'balanced'], default='balanced', 
						help="If `standard`, all classes use a weight of 1.\
							  If `balanced`, classes are weighted inverse proportionally to their size (see https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)")

	parser.add_argument('--debugging', action='store_true', dest='debugging')
	parser.set_defaults(debugging=False)
	parser.add_argument('--deterministic', action='store_true', dest='deterministic')
	parser.set_defaults(deterministic=False)

	# SEEDS
	parser.add_argument('--seed_model_init_and_training', type=int, default=42, 
		help='Seed for training and model model initializing')

	parser.add_argument('--seed_kfold', type=int, help='Seed used for doing the kfold in train/test split')
	parser.add_argument('--seed_validation', type=int, help='Seed used for selecting the validation split.')

	# Dataset loading
	parser.add_argument('--num_workers', type=int, default=1, help="number of workers for loading dataset")
	parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='dont pin memory for data loaders')
	parser.set_defaults(pin_memory=True)

	####### Wandb logging
	parser.add_argument('--logger', type=str, default='wandb', choices=['wandb', 'csv'], help='logger for logging the results')
	parser.add_argument('--experiment_name', type=str, default='', help='Name for the experiment')

	parser.add_argument('--group', type=str, help="Group runs in wand")
	parser.add_argument('--job_type', type=str, help="Job type for wand")
	parser.add_argument('--notes', type=str, help="Notes for wandb logging.")
	parser.add_argument('--tags', nargs='+', type=str, default=[], help='Tags for wandb')
	parser.add_argument('--suffix_wand_run_name', type=str, default="", help="Suffix for run name in wand")
	parser.add_argument('--wandb_log_model', action='store_true', dest='wandb_log_model',
						help='True for saving the model checkpoints in wandb')
	parser.set_defaults(wandb_log_model=False)
	parser.add_argument('--wandb_watch', choices=[None, 'parameters', 'gradients', 'all'], default=None)
	parser.add_argument('--disable_wandb', action='store_true', dest='disable_wandb',
						help='True if you dont want to crete wandb logs.')
	parser.set_defaults(disable_wandb=False)
	
	####### 模型保存控制
	parser.add_argument('--min_epochs_to_save', type=int, default=0)
	parser.add_argument('--save_top_k', type=int, default=1)

	####### Overfitting Detection
	add_overfitting_detection_args(parser)
	add_ofi_overfitting_arguments(parser)

	return parser.parse_args(args)


if __name__ == "__main__":
	warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
	try:
		warnings.filterwarnings("ignore", category=pytorch_lightning.utilities.warnings.LightningDeprecationWarning)
	except AttributeError:
		try:
			warnings.filterwarnings("ignore", category=pytorch_lightning.utilities.warnings.PossibleUserWarning)
		except AttributeError:
			warnings.filterwarnings("ignore", message=".*lightning.*")

	import warnings
	warnings.filterwarnings("ignore")

	print("Starting...")

	logging.basicConfig(
		filename=f'{BASE_DIR}/logs_exceptions.txt',
		filemode='a',
		format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
		datefmt='%H:%M:%S',
		level=logging.DEBUG
	)

	args = parse_arguments()

	# if using the MLP, then winit_first_layer_interpolation should be zero
	if args.winit_initialisation==None and args.model == 'dnn':
		args.winit_first_layer_interpolation=0

	# set seeds
	args.seed_kfold = args.repeat_id


	if args.dataset == 'prostate' or args.dataset == 'cll' or args.dataset == 'glioma':
		# `val_check_interval`` must be less than or equal to the number of the training batches
		args.val_check_interval = 4

	"""
	#### Parse dataset size
	when args.dataset=="metabric-dr__200" split into
	args.dataset = "metabric-dr"
	args.dataset_size = 200
	- 
	"""
	if "__" in args.dataset:
		args.dataset, args.dataset_size = args.dataset.split("__")
		args.dataset_size = int(args.dataset_size)


	#### Assert that the dataset is supported
	SUPPORTED_DATASETS = ['metabric-pam50', 'metabric-dr',
						  'tcga-2ysurvival', 'tcga-tumor-grade',
						  'lung', 'prostate', 'toxicity', 'cll', 'smk', 'allaml', 'gli', 'glioma']
	if args.dataset not in SUPPORTED_DATASETS:
		raise Exception(f"Dataset {args.dataset} not supported. Supported datasets are {SUPPORTED_DATASETS}")

	#### Assert sparsity parameters
	if args.sparsity_type:
		# if one of the sparsity parameters is set, then all of them must be set
		assert args.sparsity_type
		assert args.sparsity_method
		assert args.sparsity_regularizer
		# assert args.sparsity_regularizer_hyperparam

	# add best performing configuration
	if args.use_best_hyperparams:
		# if the model uses gene embeddings of any type, then use dataset specific embedding sizes.
		if args.model in ['fsnet', 'dietdnn']:
			if args.dataset=='cll':
				args.wpn_embedding_size = 70
			elif args.dataset=='lung':
				args.wpn_embedding_size = 20
			else:
				args.wpn_embedding_size = 50

		if args.sparsity_type=='global':
			if args.dataset == 'cll':
				args.sparsity_regularizer_hyperparam = 3e-4
			elif args.dataset == 'lung':
				args.sparsity_regularizer_hyperparam = 3e-5
			elif args.dataset == 'metabric-dr':
				args.sparsity_regularizer_hyperparam = 0
			elif args.dataset == 'metabric-pam50':
				args.sparsity_regularizer_hyperparam = 3e-6
			elif args.dataset == 'prostate':
				args.sparsity_regularizer_hyperparam = 3e-3
			elif args.dataset == 'smk':
				args.sparsity_regularizer_hyperparam = 3e-5
			elif args.dataset == 'tcga-2ysurvival':
				args.sparsity_regularizer_hyperparam = 3e-5
			elif args.dataset == 'tcga-tumor-grade':
				args.sparsity_regularizer_hyperparam = 3e-5
			elif args.dataset == 'toxicity':
				args.sparsity_regularizer_hyperparam = 3e-5
			
		elif args.model=='rf':
			params = {
				'cll': (3, 3),
				'lung': (3, 2),
				'metabric-dr': (7, 2),
				'metabric-pam50': (7, 2),
				'prostate': (5, 2),
				'smk': (5, 2),
				'tcga-2ysurvival': (3, 3),
				'tcga-tumor-grade': (3, 3),
				'toxicity': (5, 3)
			}

			args.rf_max_depth, args.rf_min_samples_leaf = params[args.dataset]

		elif args.model=='lgb':
			params = {
				'cll': (0.1, 2),
				'lung': (0.1, 1),
				'metabric-dr': (0.1, 1),
				'metabric-pam50': (0.01, 2),
				'prostate': (0.1, 2),
				'smk': (0.1, 2),
				'tcga-2ysurvival': (0.1, 1),
				'tcga-tumor-grade': (0.1, 1),
				'toxicity': (0.1, 2)
			}

			args.lgb_learning_rate, args.lgb_max_depth = params[args.dataset]
		
	if args.disable_wandb:
		os.environ['WANDB_MODE'] = 'disabled'

	if args.run_repeats_and_cv:
		# Run 5 fold cross-validation with 5 repeats
		args_new = dict(json.loads(json.dumps(vars(args))))

		for repeat_id in range(5):
			for test_split in range(5):
				args_new['repeat_id'] = repeat_id
				args_new['test_split'] = test_split
		
				run_experiment(argparse.Namespace(**args_new))
	else:
		run_experiment(args)