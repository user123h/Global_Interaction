from torchnmf.nmf import NMF
from _config import *
from _shared_imports import *
import scipy.io as spio
from sklearn.utils.class_weight import compute_class_weight
try:
	from pytorch_lightning.trainer.supporters import CombinedLoader
except ImportError:
    try:
        from pytorch_lightning.utilities import CombinedLoader
    except ImportError:
        CombinedLoader = None

def load_csv_data(path, labels_column=-1):
	Xy = pd.read_csv(path, index_col=0)
	X = Xy[Xy.columns[:labels_column]].to_numpy()
	y = Xy[Xy.columns[labels_column]].to_numpy()

	return X, y

def load_lung(drop_class_5=True):
	data = spio.loadmat(f'{DATA_DIR}/lung.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	if drop_class_5:
		X = X.drop(index=[156, 157, 158, 159, 160, 161])
		Y = Y.drop([156, 157, 158, 159, 160, 161])

	new_labels = {1:0, 2:1, 3:2, 4:3, 5:4}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_prostate():
	data = spio.loadmat(f'{DATA_DIR}/Prostate_GE.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_toxicity():
	data = spio.loadmat(f'{DATA_DIR}/TOX_171.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1, 3:2, 4:3}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_cll():
	data = spio.loadmat(f'{DATA_DIR}/CLL_SUB_111.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1, 3:2}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_smk():
	data = spio.loadmat(f'{DATA_DIR}/SMK_CAN_187.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_allaml():
	data = spio.loadmat(f'{DATA_DIR}/ALLAML.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_gli():
	data = spio.loadmat(f'{DATA_DIR}/GLI_85.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_glioma():
	data = spio.loadmat(f'{DATA_DIR}/GLIOMA.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1, 3:2, 4:3}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y


class CustomPytorchDataset(Dataset):
	def __init__(self, X, y, transform=None) -> None:
		# X, y are numpy
		super().__init__()

		self.X = torch.tensor(X, requires_grad=False)
		self.y = torch.tensor(y, requires_grad=False)
		self.transform = transform

	def __getitem__(self, index):
		x = self.X[index]
		y = self.y[index]
		if self.transform:
			x = self.transform(x)
			y = y.repeat(x.shape[0]) # replicate y to match the size of x

		return x, y

	def __len__(self):
		return len(self.X)


def standardize_data(X_train, X_valid, X_test, preprocessing_type):
	if preprocessing_type == 'standard':
		scaler = StandardScaler()
	elif preprocessing_type == 'minmax':
		scaler = MinMaxScaler()
	elif preprocessing_type == 'raw':
		scaler = None
	else:
		raise Exception("preprocessing_type not supported")

	if scaler:
		X_train = scaler.fit_transform(X_train).astype(np.float32)
		if X_valid is not None:
			X_valid = scaler.transform(X_valid).astype(np.float32)
		if X_test is not None:
			X_test = scaler.transform(X_test).astype(np.float32)

	return X_train, X_valid, X_test


def compute_stratified_splits(X, y, cv_folds, seed_kfold, split_id):
	skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed_kfold)
	
	for i, (train_ids, test_ids) in enumerate(skf.split(X, y)):
		if i == split_id:
			return X[train_ids], X[test_ids], y[train_ids], y[test_ids], train_ids, test_ids

def compute_histogram_embedding(args, X, embedding_size):
	X = np.rot90(X)
	
	number_features = X.shape[0]
	embedding_matrix = np.zeros(shape=(number_features, embedding_size))

	for feature_id in range(number_features):
		feature = X[feature_id]

		hist_values, bin_edges = np.histogram(feature, bins=embedding_size) # like in FsNet
		bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
		embedding_matrix[feature_id] = np.multiply(hist_values, bin_centers)

	return embedding_matrix


def compute_nmf_embeddings(Xt, rank):
	print("Approximating V = H W.T")
	print(f"Input V has shape {Xt.shape}")

	if type(Xt) != torch.Tensor:
		Xt = torch.tensor(Xt, requires_grad=False)

	nmf = NMF(Xt.shape, rank=rank).cuda()
	nmf.fit(Xt.cuda(), beta=2, max_iter=1000, verbose=True) 

	print(f"H has shape {nmf.H.shape}")
	print(f"W.T has shape {nmf.W.T.shape}")

	return nmf.W


def compute_svd_embeddings(X, rank=None):
	if type(X) != torch.Tensor:
		X = torch.tensor(X, requires_grad=False)
	assert X.shape[0] < X.shape[1]

	U, S, Vh = torch.linalg.svd(X, full_matrices=False)

	V = Vh.T

	if rank:
		S = S[:rank]
		V = V[:, :rank]

	return V, S


class DatasetModule(pl.LightningDataModule):
	def __init__(self, args, X_train, y_train, X_valid, y_valid, X_test, y_test, 
						indices_train=None, indices_valid=None, indices_test=None):
		super().__init__()
		self.args = args

		args.num_features = X_train.shape[1]
		args.num_classes = len(set(y_train).union(set(y_valid)).union(set(y_test)))

		self.indices_train = indices_train
		self.indices_valid = indices_valid
		self.indices_test = indices_test

		# Standardize data
		self.X_train_raw = X_train
		self.X_valid_raw = X_valid
		self.X_test_raw = X_test

		X_train, X_valid, X_test = standardize_data(X_train, X_valid, X_test, args.patient_preprocessing)
		
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		self.y_valid = y_valid
		self.X_test = X_test
		self.y_test = y_test

		self.train_dataset = CustomPytorchDataset(X_train, y_train)
		self.valid_dataset = CustomPytorchDataset(X_valid, y_valid)
		self.test_dataset = CustomPytorchDataset(X_test, y_test)

		self.args.train_size = X_train.shape[0]
		self.args.valid_size = X_valid.shape[0] if X_valid is not None else 0
		self.args.test_size = X_test.shape[0]
		self.graphs_dataset = None

	def train_dataloader(self):
		dataloaders = {
			"tabular": DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True, 
							num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
		}
		
		return CombinedLoader(dataloaders, mode='min_size')


	def val_dataloader(self):
		dataloaders = {
			'tabular': DataLoader(self.valid_dataset, batch_size=128, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
		}
		
		return CombinedLoader(dataloaders)

	def test_dataloader(self):
		return CombinedLoader({
			'tabular': DataLoader(self.test_dataset, batch_size=128, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
		})

	def get_embedding_matrix(self, embedding_type, embedding_size):
		"""
		Return matrix D x M

		Use a the shared hyper-parameter self.args.embedding_preprocessing.
		"""
		if embedding_type == None:
			return None
		else:
			if embedding_size == None:
				raise Exception()

		# Preprocess the data for the embeddings
		if self.args.embedding_preprocessing == 'raw':
			X_for_embeddings = self.X_train_raw
		elif self.args.embedding_preprocessing == 'standard':
			X_for_embeddings = StandardScaler().fit_transform(self.X_train_raw)
		elif self.args.embedding_preprocessing == 'minmax':
			X_for_embeddings = MinMaxScaler().fit_transform(self.X_train_raw)
		else:
			raise Exception("embedding_preprocessing not supported")

		if embedding_type == 'histogram':
			"""
			Embedding similar to FsNet
			"""
			embedding_matrix = compute_histogram_embedding(self.args, X_for_embeddings, embedding_size)
			return torch.tensor(embedding_matrix.copy(), dtype=torch.float32, requires_grad=False)
		elif embedding_type=='all_patients':
			"""
			A gene's embedding are its patients gene expressions.
			"""
			embedding_matrix = np.rot90(X_for_embeddings)[:, :embedding_size]
			return torch.tensor(embedding_matrix.copy(), dtype=torch.float32, requires_grad=False)
		elif embedding_type=='svd':
			# Vh.T (4160 x rank) contains the gene embeddings on each row
			U, S, Vh = torch.linalg.svd(torch.tensor(X_for_embeddings, dtype=torch.float32), full_matrices=False) 
			
			Vh.T.requires_grad = False
			return Vh.T[:, :embedding_size].type(torch.float32)
		elif embedding_type=='nmf':
			emd = compute_nmf_embeddings(X_for_embeddings, rank=embedding_size)
			emd_data = emd.data
			emd_data.requires_grad = False
			return emd_data.type(torch.float32)
		elif embedding_type=='zero':
			pass
		else:
			raise Exception("Invalid embedding type")


def create_data_module(args):
	if "__" in args.dataset:
		dataset, dataset_size = args.dataset.split("__")
	else:
		dataset, dataset_size = args.dataset, args.dataset_size

	if dataset=='lung':
		X, y = load_lung()
	elif dataset=='toxicity':
		X, y = load_toxicity()
	elif dataset=='prostate':
		X, y = load_prostate()
	elif dataset=='cll':
		X, y = load_cll()
	elif dataset=='smk':
		X, y = load_smk()
	elif dataset=='allaml':
		X, y = load_allaml()
	elif dataset=='gli':
		X, y = load_gli()
	elif dataset=='glioma':
		X, y = load_glioma()

	data_module = create_datamodule_with_cross_validation(args, X, y)

	if args.class_weight=='balanced':
		args.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data_module.y_train), y=data_module.y_train)
	elif args.class_weight=='standard':
		args.class_weights = compute_class_weight(class_weight=None, classes=np.unique(data_module.y_train), y=data_module.y_train)
	args.class_weights = args.class_weights.astype(np.float32)
	print(f"Weights for the classification loss: {args.class_weights}")

	return data_module


def create_datamodule_with_cross_validation(args, X, y):
	if type(X)==pd.DataFrame:
		X = X.to_numpy()
	if type(y)==pd.Series:
		y = y.to_numpy()

	assert type(X)==np.ndarray
	assert type(y)==np.ndarray

	X_train_and_valid, X_test, y_train_and_valid, y_test, indices_train_and_valid, indices_test = compute_stratified_splits(
		X, y, cv_folds=args.cv_folds, 
	seed_kfold=args.seed_kfold, split_id=args.test_split
	)

	X_train, X_valid, y_train, y_valid, indices_train, indices_valid = train_test_split(
		X_train_and_valid, y_train_and_valid, indices_train_and_valid,
		test_size = args.valid_percentage,
		random_state = args.seed_validation,
		stratify = y_train_and_valid
	)
	
	print(f"Train size: {X_train.shape[0]}\n")
	print(f"Valid size: {X_valid.shape[0]}\n")
	print(f"Test size: {X_test.shape[0]}\n")

	assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == X.shape[0]
	assert set(y_train).union(set(y_valid)).union(set(y_test)) == set(y)

	return DatasetModule(args, X_train, y_train, X_valid, y_valid, X_test, y_test,
		indices_train=indices_train, indices_valid=indices_valid, indices_test=indices_test)