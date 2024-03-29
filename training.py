import abc
import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path
from helpers.train_results import BatchResult, EpochResult, FitResult

class Trainer_1(abc.ABC):
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """
    def __init__(self, model, loss_fn, optimizer, device=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs-1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            # TODO: Train & evaluate for one epoch
            # - Use the train/test_epoch methods.
            # - Save losses and accuracies in the lists above.
            # - Optional: Implement checkpoints. You can use torch.save() to
            #   save the model to a file.
            # - Optional: Implement early stopping. This is a very useful and
            #   simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            train_result = self.train_epoch(dl_train, verbose=verbose)
            train_loss.append(torch.mean(torch.Tensor(train_result.losses)).item())
            train_acc.append(train_result.accuracy)

            test_result = self.test_epoch(dl_test, verbose=verbose)
            test_loss.append(torch.mean(torch.Tensor(test_result.losses)).item())
            test_acc.append(test_result.accuracy)

            if best_acc is None or best_acc < test_acc[-1]:
                best_acc = test_acc[-1]
                if checkpoints is not None:
                    torch.save(self.model, f'{checkpoints}_{int(os.times().elapsed)}.pt')

            if early_stopping is not None:
                if epoch == 0:
                    previous_loss = test_loss[-1]
                else:
                    current_loss = test_loss[-1]
                    print(f"test loss difference: {previous_loss - current_loss}")
                    if previous_loss - current_loss <= 0.01:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0
                    print(f"epochs_without_improvement: {epochs_without_improvement}")

                if epochs_without_improvement >= early_stopping:
                    print(f"Stopped in epoch {epoch}")
                    break
            # ========================

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                epochs_without_improvement =\
                    saved_state.get('ewi', epochs_without_improvement)
                self.model.load_state_dict(saved_state['model_state'])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            # TODO: Train & evaluate for one epoch
            # - Use the train/test_epoch methods.
            # - Save losses and accuracies in the lists above.
            # - Implement early stopping. This is a very useful and
            #   simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======

            # Self train current epoch.
            train_result = self.train_epoch(dl_train, verbose=verbose)
            train_loss.extend(train_result.losses)
            train_acc.append(train_result.accuracy)

            test_result = self.test_epoch(dl_test, verbose=verbose)
            test_loss.extend(test_result.losses)
            test_acc.append(test_result.accuracy)

            if (not best_acc) or (best_acc < test_acc[-1]):
                best_acc = test_acc[-1]
                # Need to save checkpoint here.
                save_checkpoint = True
                is_checkpoint_saved = True

            import numpy as np
            if epoch == 0:
                c_loss = np.mean(test_result.losses)
            else:
                prev_loss = c_loss
                c_loss = np.mean(test_result.losses)
                if c_loss < prev_loss:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            # Check early stopping.
            if early_stopping and epochs_without_improvement >= early_stopping:
                actual_num_epochs = epoch
                break
            # ========================

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.state_dict())
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch+1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)


class RNNTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_epoch(self, dl_train: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        self.hidden_state = None
        # ========================
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        self.hidden_state = None
        # ========================
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        # TODO: Train the RNN model on one batch of data.
        # - Forward pass
        # - Calculate total loss over sequence
        # - Backward pass (BPTT)
        # - Update params
        # - Calculate number of correct char predictions
        # ====== YOUR CODE: ======
        y_scores, self.hidden_state = self.model(x, hidden_state=self.hidden_state)

        y_scores = torch.transpose(y_scores, 1, 2)

        # Set loss function.
        loss = self.loss_fn(y_scores, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Handle hidden state.
        self.hidden_state = self.hidden_state.detach()
        self.hidden_state.requires_grad = False

        # set pred and sum to gen num_correct.
        y_predictions = torch.argmax(y_scores, dim=1)
        num_correct = torch.sum(y == y_predictions)
        # ========================

        # Note: scaling num_correct by seq_len because each sample has seq_len
        # different predictions.
        return BatchResult(loss.item(), num_correct.item() / seq_len)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        with torch.no_grad():
            # TODO: Evaluate the RNN model on one batch of data.
            # - Forward pass
            # - Loss calculation
            # - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            y_scores_list, self.hidden_state = self.model(x, hidden_state=self.hidden_state)

            # Make it work for CrossEntropy.
            y_scores_list = torch.transpose(y_scores_list, 1, 2)
            loss = self.loss_fn(y_scores_list, y)

            y_predictions = torch.argmax(y_scores_list, dim=1)

            num_correct = torch.sum(y == y_predictions)
            # ========================

        return BatchResult(loss.item(), num_correct.item() / seq_len)


class VAETrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)
        # TODO: Train a VAE on one batch.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return BatchResult(loss.item(), 1/data_loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)

        with torch.no_grad():
            # TODO: Evaluate a VAE on one batch.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================

        return BatchResult(loss.item(), 1/data_loss.item())


class BlocksTrainer(Trainer_1):
    def __init__(self, model, loss_fn, optimizer):
        super().__init__(model, loss_fn, optimizer)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch

        # TODO: Train the Block model on one batch of data.
        # - Forward pass
        # - Backward pass
        # - Optimize params
        # - Calculate number of correct predictions
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch

        # TODO: Evaluate the Block model on one batch of data.
        # - Forward pass
        # - Calculate number of correct predictions
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return BatchResult(loss, num_correct)


class TorchTrainer(Trainer_1):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        # TODO: Train the PyTorch model on one batch of data.
        # - Forward pass
        # - Backward pass
        # - Optimize params
        # - Calculate number of correct predictions
        # ====== YOUR CODE: ======
        self.optimizer.zero_grad()
        x_scores = self.model.forward(X)
        loss = self.loss_fn(x_scores, y)
        loss.backward()
        self.optimizer.step()
        _, y_predicted = torch.max(x_scores, dim=1)
        num_correct = (y_predicted == y).sum().item()
        # ========================

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            # TODO: Evaluate the PyTorch model on one batch of data.
            # - Forward pass
            # - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            x_scores = self.model(X)
            loss = self.loss_fn(x_scores, y)
            _, y_predicted = torch.max(x_scores, 1)
            num_correct = (y_predicted == y).sum().item()
            # ========================

        return BatchResult(loss, num_correct)