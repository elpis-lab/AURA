from tqdm import tqdm
import numpy as np
import copy
from typing import Callable
import torch
from torch.utils.data import DataLoader, TensorDataset


class TorchModel:
    """A wrapper class for torch model"""

    def __init__(
        self,
        model_class: torch.nn.Module,  # model class, not the instance
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 1e-3,
        batch_size: int = 16,
        epochs: int = 1000,
        loss_fn: Callable = None,
        score_fn: Callable = None,
        verbose: int = 0,
        require_training: bool = True,
        device: torch.device = None,
    ):
        """Initialize the wrapper"""
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_class = model_class

        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.score_fn = score_fn if score_fn is not None else loss_fn
        self.verbose = verbose
        self.require_training = require_training

    def fit(self, x, y, val_x=None, val_y=None, save_best=None, **kwargs):
        """Train the model with given data"""
        self.model = self.model_class(**kwargs).to(self.device)
        if not self.require_training:
            return [], []

        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True)
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=20, factor=0.5
        )

        # initialize tracking
        if save_best is not None:
            if save_best == "train" or save_best == "val":
                if save_best == "val" and (val_x is None or val_y is None):
                    raise ValueError("No validation data provided but save_best is 'val'")
                best_loss = np.inf
                best_weights = copy.deepcopy(self.model.state_dict())
            else:
                raise ValueError(f"save_best must be 'train' or 'val'")
        losses, val_losses = [], []

        if self.verbose:
            p_bar = tqdm(range(self.epochs))
        else:
            p_bar = range(self.epochs)
        for e in p_bar:
            # Training
            self.model.train()
            epoch_loss = 0
            epoch_val_loss = 0

            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(x, **kwargs)
                loss = self.loss_fn(y_pred, y, **kwargs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x)

            epoch_loss /= len(dataloader.dataset)
            scheduler.step(epoch_loss)

            # Validation
            self.model.eval()
            if val_x is not None and val_y is not None:
                epoch_val_loss = self.score(val_x, val_y, **kwargs)

            # Record
            losses.append(epoch_loss)
            val_losses.append(epoch_val_loss)

            # Save best model
            metric = epoch_val_loss if save_best == "val" else epoch_loss
            if save_best and metric < best_loss:
                best_loss = metric
                best_weights = copy.deepcopy(self.model.state_dict())

            # Verbose
            if self.verbose:
                p_bar.set_description(
                    f"EPOCH {e+1} | Loss: {epoch_loss:.4f}"
                    + f"| Val Loss: {epoch_val_loss:.4f}"
                    + f"| LR: {scheduler.get_last_lr()[0]:.4f}"
                )

        # Restore best model
        if save_best:
            self.model.load_state_dict(best_weights)

        return losses, val_losses

    def predict(self, x, **kwargs):
        """Predict the output with given input"""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.dim() < 2:
            x = x.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            y = self.model(x, **kwargs)
        return y.cpu().numpy()

    def __call__(self, x, **kwargs):
        """Call the model"""
        return self.predict(x, **kwargs)

    def score(self, x, y, **kwargs):
        """Calculate the score with given data"""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            pred_y = self.model(x, **kwargs)

        score = self.score_fn(pred_y, y).item()
        return score

    def save(self, path):
        """Save model"""
        if not self.model:
            print("Model is not trained yet")
            return
        torch.save(self.model.state_dict(), path)

    def load(self, path, **kwargs):
        """Load model"""
        self.model = self.model_class(**kwargs).to(self.device)
        self.model.load_state_dict(
            torch.load(path, weights_only=True, map_location=self.device)
        )


class OptTorchModel:
    """
    A class that uses torch model for optimization
    The output (can be random or by another model) gets optimized by a
    given model with its gradient.
    """

    def __init__(
        self,
        model,  # model that provides forward prediction
        start_model,  # model that provides initial guess
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 1e-3,
        epochs: int = 1000,
        verbose: int = 0,
        device: torch.device = None,
    ):
        """Initialize the wrapper"""
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.start_model = start_model
        if isinstance(self.model, torch.nn.Module):
            self.model = self.model.to(self.device)
        if isinstance(self.start_model, torch.nn.Module):
            self.start_model = self.start_model.to(self.device)

        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, x, y, val_x=None, val_y=None, **kwargs):
        """Train the model with given data"""
        # This model uses existing model, so no need to train
        return [], []

    def predict(self, y, x_min=None, x_max=None, **kwargs):
        """Predict the output with desired output y"""
        goal_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Initialize the starting point, Shape = (n_sample, n_start, n_feature)
        start_x = self.start_model(goal_tensor)[:, :3]
        # in case model only provides 1 start point
        if start_x.dim() == 2:
            start_x = start_x.unsqueeze(1)
        n_sample, n_start, n_feature = start_x.shape

        # For clamping the x later
        if x_min is not None and x_max is not None:
            x_min = torch.tensor(x_min, dtype=torch.float32, device=self.device).view(1, n_feature)
            x_max = torch.tensor(x_max, dtype=torch.float32, device=self.device).view(1, n_feature)

        # Change the tensor shape for later computation
        goal_tensor = goal_tensor.view(n_sample, 1, n_feature)
        start_x = start_x.view(n_sample * n_start, n_feature)
        if x_min is not None and x_max is not None:
            start_x = torch.clamp(start_x, x_min, x_max)

        # Start optimization start_x
        start_x = start_x.detach().clone().requires_grad_(True)
        optimizer = self.optimizer([start_x], lr=self.lr)

        if self.verbose:
            p_bar = tqdm(range(self.epochs))
        else:
            p_bar = range(self.epochs)
        self.model.eval()
        for _ in p_bar:

            optimizer.zero_grad()
            y_pred = self.model(start_x)[:, :3]
            # y_pred with shape (n_sample, n_start, n_feature),
            # goal_tensor with shape (n_sample, 1, n_feature).
            # all_loss with shape (n_sample, n_start)
            all_loss = torch.mean(
                (y_pred.view(n_sample, n_start, n_feature) - goal_tensor) ** 2,
                dim=2,
            )
            loss = all_loss.mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if x_min is not None and x_max is not None:
                    start_x.data = torch.clamp(start_x, min=x_min, max=x_max)

        # Find the best loss of each sample from all_loss
        start_x = start_x.detach().view(n_sample, n_start, n_feature)
        _, best_indices = torch.min(all_loss, dim=1)
        sample_indices = torch.arange(n_sample, device=start_x.device)
        best_x = start_x[sample_indices, best_indices, :]
        return best_x.cpu().numpy()

    def score(self, x, y, **kwargs):
        """Calculate the score with given data"""
        # This model uses existing model, so no need to score
        return 0

    def predict_y(self, x, **kwargs):
        """Just a wrapper function to predict y with model"""
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y = self.model(x_tensor, **kwargs)
            y = y.cpu().numpy()
        return y


class OptTorchModel_2:
    """A class that uses torch model for optimization
    The output (can be random or by another model) gets optimized by a
    given model with its gradient.
    """

    def __init__(
        self,
        model,  # model that provides forward output
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 1e-4,
        epochs: int = 1000,
        verbose: int = 0,
        device: torch.device = None,
        scheduler_patience: int = 100,
        scheduler_factor: float = 0.1,
        min_lr: float = 1e-6,
    ):
        """Initialize the wrapper"""
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        if isinstance(self.model, torch.nn.Module):
            self.model = self.model.to(self.device)

        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.min_lr = min_lr

    def fit(self, x, y, val_x=None, val_y=None, **kwargs):
        """Train the model with given data"""
        # This model uses existing model, so no need to train
        return [], []

    def predict(self, y, start_guess, x_min=None, x_max=None, plot=True, **kwargs):
        """Predict the output with desired output y"""
        import matplotlib.pyplot as plt

        goal_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Check if goal_tensor is empty
        if goal_tensor.numel() == 0:
            raise ValueError("goal_tensor is empty. Cannot run optimization.")

        # Initialize the starting point, Shape = (n_sample, n_start, n_feature)
        if isinstance(start_guess, np.ndarray):
            start_x = torch.tensor(start_guess, dtype=torch.float32).to(self.device)
        else:
            start_x = start_guess.to(self.device)  # Move to same device as model

        # Check if start_x is empty
        if start_x.numel() == 0:
            raise ValueError("start_guess is empty. Cannot run optimization.")

        # in case model only provides 1 start point
        if start_x.dim() == 2:
            start_x = start_x.unsqueeze(1)
        n_sample, n_start, n_feature = start_x.shape

        # Check if feature dimension is 0
        if n_feature == 0:
            raise ValueError(
                f"Feature dimension is 0. start_x shape: {start_x.shape}, goal_tensor shape: {goal_tensor.shape}"
            )

        # For clamping the x later
        if x_min is not None and x_max is not None:
            x_min = torch.tensor(x_min, dtype=torch.float32, device=self.device).view(1, n_feature)
            x_max = torch.tensor(x_max, dtype=torch.float32, device=self.device).view(1, n_feature)

        # Change the tensor shape for later computation
        # Ensure goal_tensor has the correct last dimension
        if goal_tensor.dim() == 1:
            goal_tensor = goal_tensor.unsqueeze(0)

        if goal_tensor.shape[-1] != n_feature:
            # If goal_tensor doesn't match, reshape it
            if goal_tensor.numel() % n_feature != 0:
                raise ValueError(
                    f"Cannot reshape goal_tensor with {goal_tensor.numel()} elements into shape with feature dimension {n_feature}"
                )
            goal_tensor = goal_tensor.view(-1, n_feature)

        goal_tensor = goal_tensor.view(
            -1, 1, goal_tensor.shape[-1]
        )  # Reshape to (n_sample, 1, n_feature)
        start_x = start_x.view(n_sample * n_start, n_feature)
        if x_min is not None and x_max is not None:
            start_x = torch.clamp(start_x, x_min, x_max)

        # Start optimization start_x
        start_x = start_x.detach().clone().requires_grad_(True)
        optimizer = self.optimizer([start_x], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.min_lr,
            # verbose=self.verbose > 0,
        )

        # Track losses and best results
        losses = []
        learning_rates = []  # Track learning rate history
        best_loss = float("inf")
        best_x = None
        no_improvement_count = 0
        max_no_improvement = 200  # Early stopping patience

        if self.verbose:
            p_bar = tqdm(range(self.epochs))
        else:
            p_bar = range(self.epochs)
        self.model.eval()
        for epoch in p_bar:
            optimizer.zero_grad()
            y_pred = self.model(start_x)[:, :3]
            # y_pred with shape (n_sample, n_start, n_feature),
            # goal_tensor with shape (n_sample, 1, n_feature).
            # all_loss with shape (n_sample, n_start)
            all_loss = torch.mean(
                (y_pred.view(n_sample, n_start, n_feature) - goal_tensor) ** 2,
                dim=2,
            )
            loss = all_loss.mean()
            losses.append(loss.item())  # Track loss

            # Update best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_x = start_x.detach().clone()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= max_no_improvement:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if x_min is not None and x_max is not None:
                    start_x.data = torch.clamp(start_x, min=x_min, max=x_max)

            # Update learning rate
            scheduler.step(loss)
            learning_rates.append(
                scheduler._last_lr[0]
                if hasattr(scheduler, "_last_lr")
                else optimizer.param_groups[0]["lr"]
            )

            # Update progress bar if verbose
            if self.verbose:
                p_bar.set_description(f"Loss: {loss.item():.4f} | LR: {scheduler._last_lr[0]:.2e}")

        # Plot the loss curve (save to file when plot=False, show when plot=True)
        # if plot or len(losses) > 0:  # Always create the plot
        #     import matplotlib

        #     if not plot:
        #         matplotlib.use(
        #             "Agg"
        #         )  # Use non-interactive backend for file saving

        #     plt.figure(figsize=(10, 5))
        #     plt.plot(losses)
        #     plt.title("Optimization Loss Over Time")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Loss")
        #     plt.grid(True)

        #     if plot:
        #         plt.show()
        # else:
        #     # Save to file with timestamp
        #     import datetime

        #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #     filename = f"optimization_loss_{timestamp}.png"
        #     plt.savefig(filename, dpi=150, bbox_inches="tight")
        #     print(f"     - Loss plot saved to: {filename}")
        #     plt.close()  # Close the figure to free memory

        # Use best result found during optimization
        if best_x is not None:
            start_x = best_x

        # Find the best loss of each sample from all_loss
        start_x = start_x.view(n_sample, n_start, n_feature)
        _, best_indices = torch.min(all_loss, dim=1)
        sample_indices = torch.arange(n_sample, device=start_x.device)
        best_x = start_x[sample_indices, best_indices, :]

        # Final clamping to ensure values stay within bounds
        if x_min is not None and x_max is not None:
            best_x = torch.clamp(best_x, min=x_min, max=x_max)

        # Save loss history to instance for external access
        self.loss_history = losses
        self.learning_rate_history = learning_rates

        return best_x.cpu().numpy(), best_loss

    def score(self, x, y, **kwargs):
        """Calculate the score with given data"""
        # This model uses existing model, so no need to score
        return 0

    def predict_y(self, x, **kwargs):
        """Just a wrapper function to predict y with model"""
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y = self.model(x_tensor, **kwargs)
            y = y.cpu().numpy()
        return y
