import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import logging
import numpy as np


class OFIOverfittingCallback(Callback):
    def __init__(self, patience=3, threshold=0.1, min_train_acc=0.9, verbose=True, max_acc_threshold=0.25, min_epochs=5):
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.min_train_acc = min_train_acc
        self.verbose = verbose
        self.max_acc_threshold = max_acc_threshold
        self.min_epochs = min_epochs

        self.alpha, self.beta, self.gamma = 1.0, 0.5, 0.5
        self.window_size = 5
        self.dynamic_factor = 0.3

        self.ofi_history = []
        self.val_acc_history = []
        self.consecutive_count = 0

        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[OFI] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.logged_metrics

        train_acc = logs.get("train/balanced_accuracy")
        val_acc = logs.get("val/balanced_accuracy")
        train_loss = logs.get("train/total_loss")
        val_loss = logs.get("val/total_loss")

        if None in [train_acc, val_acc, train_loss, val_loss]:
            if self.verbose:
                self.logger.warning("Essential indicators are missing; OFI cannot be calculated.")
            return

        train_acc = float(train_acc)
        val_acc = float(val_acc)
        train_loss = float(train_loss)
        val_loss = float(val_loss)

        gap_acc = train_acc - val_acc
        gap_loss = val_loss - train_loss

        self.val_acc_history.append(val_acc)
        if len(self.val_acc_history) > self.window_size:
            self.val_acc_history.pop(0)

        if len(self.val_acc_history) >= 3: 
            x = np.arange(len(self.val_acc_history))
            y = np.array(self.val_acc_history)

            slope = np.polyfit(x, y, 1)[0]
        elif len(self.val_acc_history) > 1:
  
            slope = (self.val_acc_history[-1] - self.val_acc_history[0]) / (len(self.val_acc_history) - 1)
        else:
            slope = 0.0

 
        ofi = self.alpha * gap_acc + self.beta * gap_loss + self.gamma * slope
        self.ofi_history.append(ofi)


        if len(self.ofi_history) < 3: 
            threshold = float('inf') 
        else:
            threshold = np.mean(self.ofi_history) + self.dynamic_factor * np.std(self.ofi_history)

        if self.verbose:
            self.logger.info(f"Epoch {trainer.current_epoch} - OFI: {ofi:.6f}, Threshold: {threshold:.6f}")
            self.logger.info(f"  gap_acc={gap_acc:.6f}, gap_loss={gap_loss:.6f}, slope={slope:.6f}")


        current_epoch = trainer.current_epoch
        
  
        epoch_protection = current_epoch < self.min_epochs
        acc_protection = train_acc < self.min_train_acc
        
        if epoch_protection or acc_protection:
            if self.verbose and (epoch_protection or acc_protection):
                protection_reasons = []
                if epoch_protection:
                    protection_reasons.append(f"Round protection({current_epoch}<{self.min_epochs})")
                if acc_protection:
                    protection_reasons.append(f"Accuracy protection({train_acc:.3f}<{self.min_train_acc})")
                self.logger.info(f"The protection mechanism is now active.: {', '.join(protection_reasons)}")
            self.consecutive_count = 0
            return
        
        if ofi > threshold:
            self.consecutive_count += 1
            if self.verbose:
                self.logger.warning(f"Overfitting signal detected! Consecutive {self.consecutive_count}/{self.patience}")
            if self.consecutive_count >= self.patience:
                self.logger.error("="*60)
                self.logger.error("OFI overfitting detection triggered; training stopped!")
                self.logger.error(f"Final metrics: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
                self.logger.error(f"OFI={ofi:.6f}, threshold={threshold:.6f}")
                self.logger.error("="*60)
                trainer.should_stop = True
        else:
            if self.consecutive_count > 0:
                if self.verbose:
                    self.logger.info("Overfitting signal gone, counter reset.")
            self.consecutive_count = 0

    def on_train_end(self, trainer, pl_module):
        if trainer.should_stop and self.consecutive_count >= self.patience:
            if self.verbose:
                self.logger.info("Training was early-stopped due to OFI overfitting detection.")
        else:
            if self.verbose:
                self.logger.info("Training completed normally; no early stop triggered.")
