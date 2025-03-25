# import torch
# from tqdm import tqdm
# import mlflow
# import os
# import pickle
# from ..config import config

# class Trainer:
#     def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device):
#         self.model = model
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer
#         self.device = device
        
#         # Create models directory if it doesn't exist
#         self.model_save_path = config.MODEL_DIR
#         os.makedirs(self.model_save_path, exist_ok=True)

#     def train_step(self):
#         self.model.train()
#         train_loss, train_acc = 0, 0
        
#         for batch, (X, y) in enumerate(self.train_loader):
#             X, y = X.to(self.device), y.to(self.device)
            
#             out = self.model(X.float())
#             loss = self.loss_fn(out, y)
#             train_loss += loss.item()

#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             preds = torch.argmax(out, dim=1)
#             train_acc += (preds == y).sum().item() / len(y)

#         return train_loss / len(self.train_loader), train_acc / len(self.train_loader)

#     def val_step(self):
#         self.model.eval()
#         val_loss, val_acc = 0, 0

#         with torch.inference_mode():
#             for X, y in self.val_loader:
#                 X, y = X.to(self.device), y.to(self.device)
                
#                 out = self.model(X.float())
#                 loss = self.loss_fn(out, y)
#                 val_loss += loss.item()

#                 preds = torch.argmax(out, dim=1)
#                 val_acc += (preds == y).sum().item() / len(y)
        
#         return val_loss / len(self.val_loader), val_acc / len(self.val_loader)

#     def save_model(self, epoch, val_acc):
#         """Save model as pickle file"""
#         model_filename = f"model_epoch_{epoch}_acc_{val_acc:.3f}.pkl"
#         model_path = os.path.join(self.model_save_path, model_filename)
        
#         # Save as pickle
#         with open(model_path, 'wb') as f:
#             pickle.dump(self.model, f)
        
#         return model_path

#     def train(self, epochs):
#         # Set up MLflow tracking
#         mlflow.set_tracking_uri("file:" + str(config.ROOT_DIR / "mlruns"))
#         experiment_name = "product_classification"
#         mlflow.set_experiment(experiment_name)

#         best_val_acc = 0
#         results = {
#             "train_loss": [], "train_acc": [],
#             "test_loss": [], "test_acc": []
#         }

#         # Start MLflow run
#         with mlflow.start_run(run_name=f"training_{config.MODEL_NAME}"):
#             # Log parameters
#             mlflow.log_params({
#                 "model_name": config.MODEL_NAME,
#                 "epochs": epochs,
#                 "batch_size": config.BATCH_SIZE,
#                 "learning_rate": config.OPTIMIZER["params"]["lr"],
#                 "weight_decay": config.OPTIMIZER["params"]["weight_decay"],
#                 "image_size": config.IMAGE_SIZE
#             })

#             print("Starting training...")
#             for epoch in tqdm(range(epochs), desc="Training Progress"):
#                 train_loss, train_acc = self.train_step()
#                 val_loss, val_acc = self.val_step()

#                 # Store results
#                 results["train_loss"].append(train_loss)
#                 results["train_acc"].append(train_acc)
#                 results["test_loss"].append(val_loss)
#                 results["test_acc"].append(val_acc)

#                 # Log metrics to MLflow
#                 mlflow.log_metrics({
#                     "train_loss": train_loss,
#                     "train_accuracy": train_acc,
#                     "val_loss": val_loss,
#                     "val_accuracy": val_acc
#                 }, step=epoch)

#                 print(f"Epoch {epoch+1}/{epochs} | "
#                       f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | "
#                       f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")

#                 # Save best model
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     model_path = self.save_model(epoch + 1, val_acc)
#                     mlflow.log_artifact(model_path, "models")
#                     print(f"Saved best model with validation accuracy: {val_acc:.3f}")

#             # Log final results plot
#             self._log_results_plot(results)

#         return results

#     def _log_results_plot(self, results):
#         """Create and log training results plot"""
#         import matplotlib.pyplot as plt

#         plt.figure(figsize=(10, 5))
        
#         # Plot training & validation accuracy
#         plt.subplot(1, 2, 1)
#         plt.plot(results["train_acc"], label="Train Accuracy")
#         plt.plot(results["test_acc"], label="Val Accuracy")
#         plt.title("Model Accuracy")
#         plt.xlabel("Epoch")
#         plt.ylabel("Accuracy")
#         plt.legend()
        
#         # Plot training & validation loss
#         plt.subplot(1, 2, 2)
#         plt.plot(results["train_loss"], label="Train Loss")
#         plt.plot(results["test_loss"], label="Val Loss")
#         plt.title("Model Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.legend()
        
#         # Save plot
#         plot_path = os.path.join(self.model_save_path, "training_results.png")
#         plt.savefig(plot_path)
#         plt.close()
        
#         # Log plot to MLflow
#         mlflow.log_artifact(plot_path, "plots")


# # final version
# import torch
# from tqdm import tqdm
# import mlflow
# import os
# import pickle
# #from ..config import config
# # from ..utils.env_loader import load_mlflow_env

# class Trainer:
#     def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device):
#         self.model = model
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer
#         self.device = device
        
#         # Create models directory if it doesn't exist
#         self.model_save_path = config.MODEL_DIR
#         os.makedirs(self.model_save_path, exist_ok=True)
        
#         # Load MLflow credentials from .env
#         # load_mlflow_env()

#     def train_step(self):
#         self.model.train()
#         train_loss, train_acc = 0, 0
        
#         for batch, (X, y) in enumerate(self.train_loader):
#             X, y = X.to(self.device), y.to(self.device)
            
#             out = self.model(X.float())
#             loss = self.loss_fn(out, y)
#             train_loss += loss.item()

#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             preds = torch.argmax(out, dim=1)
#             train_acc += (preds == y).sum().item() / len(y)

#         return train_loss / len(self.train_loader), train_acc / len(self.train_loader)

#     def val_step(self):
#         self.model.eval()
#         val_loss, val_acc = 0, 0

#         with torch.inference_mode():
#             for X, y in self.val_loader:
#                 X, y = X.to(self.device), y.to(self.device)
                
#                 out = self.model(X.float())
#                 loss = self.loss_fn(out, y)
#                 val_loss += loss.item()

#                 preds = torch.argmax(out, dim=1)
#                 val_acc += (preds == y).sum().item() / len(y)
        
#         return val_loss / len(self.val_loader), val_acc / len(self.val_loader)

#     def save_model(self, epoch, val_acc):
#         """Save model as pickle file"""
#         model_filename = f"model_epoch_{epoch}_acc_{val_acc:.3f}.pkl"
#         model_path = os.path.join(self.model_save_path, model_filename)
        
#         with open(model_path, 'wb') as f:
#             pickle.dump(self.model, f)
        
#         return model_path

#     def train(self, epochs):
#         # Set up MLflow experiment
#         experiment_name = "product_classification"
#         mlflow.set_experiment(experiment_name)

#         best_val_acc = 0
#         results = {
#             "train_loss": [], "train_acc": [],
#             "test_loss": [], "test_acc": []
#         }

#         # Start MLflow run
#         with mlflow.start_run(run_name=f"training_{config.MODEL_NAME}") as run:
#             # Log parameters
#             mlflow.log_params({
#                 "model_name": config.MODEL_NAME,
#                 "epochs": epochs,
#                 "batch_size": config.BATCH_SIZE,
#                 "learning_rate": config.OPTIMIZER["params"]["lr"],
#                 "weight_decay": config.OPTIMIZER["params"]["weight_decay"],
#                 "image_size": config.IMAGE_SIZE,
#                 "num_classes": len(config.CHOSEN_SUBCATEGORIES)
#             })

#             print("Starting training...")
#             for epoch in tqdm(range(epochs), desc="Training Progress"):
#                 train_loss, train_acc = self.train_step()
#                 val_loss, val_acc = self.val_step()

#                 # Store results
#                 results["train_loss"].append(train_loss)
#                 results["train_acc"].append(train_acc)
#                 results["test_loss"].append(val_loss)
#                 results["test_acc"].append(val_acc)

#                 # Log metrics to MLflow
#                 mlflow.log_metrics({
#                     "train_loss": train_loss,
#                     "train_accuracy": train_acc,
#                     "val_loss": val_loss,
#                     "val_accuracy": val_acc
#                 }, step=epoch)

#                 print(f"Epoch {epoch+1}/{epochs} | "
#                       f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | "
#                       f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")

#                 # Save best model
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     model_path = self.save_model(epoch + 1, val_acc)
#                     mlflow.log_artifact(model_path, "models")
#                     print(f"Saved best model with validation accuracy: {val_acc:.3f}")

#             # Log final results plot
#             self._log_results_plot(results)

#             print(f"MLflow Run ID: {run.info.run_id}")

#         return results

#     def _log_results_plot(self, results):
#         """Create and log training results plot"""
#         import matplotlib.pyplot as plt

#         plt.figure(figsize=(10, 5))
        
#         # Plot training & validation accuracy
#         plt.subplot(1, 2, 1)
#         plt.plot(results["train_acc"], label="Train Accuracy")
#         plt.plot(results["test_acc"], label="Val Accuracy")
#         plt.title("Model Accuracy")
#         plt.xlabel("Epoch")
#         plt.ylabel("Accuracy")
#         plt.legend()
        
#         # Plot training & validation loss
#         plt.subplot(1, 2, 2)
#         plt.plot(results["train_loss"], label="Train Loss")
#         plt.plot(results["test_loss"], label="Val Loss")
#         plt.title("Model Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.legend()
        
#         # Save plot
#         plot_path = os.path.join(self.model_save_path, "training_results.png")
#         plt.savefig(plot_path)
#         plt.close()
        
#         # Log plot to MLflow
#         mlflow.log_artifact(plot_path, "plots")