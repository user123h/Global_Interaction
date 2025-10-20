
import argparse
from overfitting_callback import OFIOverfittingCallback
from pytorch_lightning.callbacks import EarlyStopping


def add_overfitting_detection_args(parser):
    """
    Add command line arguments for standard overfitting detection (EarlyStopping)
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to
    """
    group = parser.add_argument_group('Overfitting Detection')
    
    group.add_argument('--enable_overfitting_detection', action='store_true',
                      help='Enable standard overfitting detection using EarlyStopping')
    
    group.add_argument('--overfitting_monitor', type=str, default='val_loss',
                      help='Metric to monitor for overfitting detection')
    
    group.add_argument('--overfitting_patience', type=int, default=5,
                      help='Number of epochs with no improvement after which training will be stopped')
    
    group.add_argument('--overfitting_min_delta', type=float, default=0.001,
                      help='Minimum change to qualify as an improvement')
    
    group.add_argument('--overfitting_mode', type=str, default='min', choices=['min', 'max'],
                      help='Whether to minimize or maximize the monitored quantity')
    
    group.add_argument('--overfitting_verbose', type=int, default=1,
                      help='Verbosity level for overfitting detection')


def add_ofi_overfitting_arguments(parser):
    """
    Add OFI overfitting detection arguments to the argument parser
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to
    """
    group = parser.add_argument_group('OFI Overfitting Detection')
    
    group.add_argument('--enable_ofi_overfitting_detection', action='store_true',
                      help='Enable OFI-based overfitting detection')
    
    group.add_argument('--ofi_patience', type=int, default=3,
                      help='Number of consecutive epochs above threshold before stopping')
    
    group.add_argument('--ofi_threshold', type=float, default=0.1,
                      help='OFI threshold for overfitting detection')
    
    group.add_argument('--ofi_min_train_acc', type=float, default=0.9,
                      help='Minimum training accuracy before OFI detection can trigger')
    
    group.add_argument('--ofi_verbose', type=int, default=1,
                      help='Verbosity level for OFI overfitting detection')
    
    group.add_argument('--ofi_max_acc_threshold', type=float, default=0.25,
                      help='Maximum allowed difference between train and validation accuracy')
    
    group.add_argument('--ofi_min_epochs', type=int, default=5,
                      help='Minimum number of epochs before OFI detection can trigger')


def create_overfitting_callback(args):
    """
    Create standard overfitting detection callback based on arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        EarlyStopping callback if enabled, None otherwise
    """
    if not getattr(args, 'enable_overfitting_detection', False):
        return None
    
    return EarlyStopping(
        monitor=getattr(args, 'overfitting_monitor', 'val_loss'),
        patience=getattr(args, 'overfitting_patience', 10),
        min_delta=getattr(args, 'overfitting_min_delta', 0.001),
        mode=getattr(args, 'overfitting_mode', 'min'),
        verbose=getattr(args, 'overfitting_verbose', 1) > 0
    )


def create_ofi_overfitting_callback(args):
    """
    Create OFI (Overfitting Index) overfitting detection callback based on arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        OFIOverfittingCallback if enabled, None otherwise
    """
    if not getattr(args, 'enable_ofi_overfitting_detection', False):
        return None
    
    return OFIOverfittingCallback(
        patience=getattr(args, 'ofi_patience', 3),
        threshold=getattr(args, 'ofi_threshold', 0.1),
        min_train_acc=getattr(args, 'ofi_min_train_acc', 0.9),
        verbose=getattr(args, 'ofi_verbose', 1) > 0,
        max_acc_threshold=getattr(args, 'ofi_max_acc_threshold', 0.25),
        min_epochs=getattr(args, 'ofi_min_epochs', 5)
    ) 