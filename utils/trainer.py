import torch
import os

class Trainer():
    def __init__(self, model, train_data_loader, test_data_loader,
               val_data_loader, loss_fn, optimizer, device, scheduler,
               forecasting_mode, init_steps=10, save_path='', ckpt_path=None):
        """
        Args:
          model: model to train
          train_data_loader: train data loader
          test_data_loader: test data loader
          val_data_loader: validation data loader
          loss_fn: loss function
          optimizer: optimizer
          device: device
          forecasting_mode: either 'one_step' or 'multi_step'
          init_steps: number of initial steps for multi-step forecasting
        """
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.forecasting_mode = forecasting_mode
        self.init_steps = init_steps

        self.save_path = save_path
        self.ckpt_path = ckpt_path
        
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self.load_model(ckpt_path)
            
        self.model.to(device)
        self.scheduler = scheduler

        self.train_loss = []
        self.test_loss = []
        self.val_loss = []

    def prepare_data(self, data):
        """
        Prepare input and target based on forecasting mode.
        data: (Batch, Time, Channel) or (Batch, Time, Channel, Feature)
        """
        # Note: data might be 3D (B, T, C) if use_graph=False in dataset
        if self.forecasting_mode == 'one_step':
            input_data = data[:, :-1, :]
            target_data = data[:, 1:, :]
        elif self.forecasting_mode == 'multi_step':
            future_step = data.shape[1] - self.init_steps
            # masking out future dataset
            
            # Use the last known step to fill the future input (autoregressive placeholder style)
            # data is tensor
            
            input_data = torch.cat([
                data[:, :self.init_steps], 
                torch.repeat_interleave(
                    data[:, self.init_steps-1:self.init_steps],
                    future_step, dim=1
                )], dim=1)
                
            target_data = data[:, self.init_steps:]
        else:
            raise ValueError('forecasting_mode must be either one_step or multi_step')
            
        return input_data, target_data

    def loss_function(self, prediction, target):
        if self.forecasting_mode == 'one_step':
            loss = self.loss_fn(prediction, target)
        else:
            # Multi-step: compare prediction steps with target steps
            # prediction shape: (B, T, C)
            # target shape: (B, T-init_steps, C)
            # prediction should be sliced to match target time steps if model outputs full sequence
            
            # Based on demo logic:
            # loss = self.loss_fn(prediction[:, self.init_steps:], target)
            loss = self.loss_fn(prediction[:, self.init_steps:], target)
        return loss

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch in self.train_data_loader:
                batch = batch.to(self.device).float() # Ensure float
                input_data, target_data = self.prepare_data(batch)
                
                self.optimizer.zero_grad()
                output = self.model(input_data)
                
                loss = self.loss_function(output, target_data)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Step scheduler if it exists
            if self.scheduler:
                self.scheduler.step()

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.model.eval()
                val_loss = self.validation()
                self.model.train()
                print(f'Epoch {epoch}, Train Loss: {train_loss / len(self.train_data_loader):.6f}, Val Loss:{val_loss / len(self.val_data_loader):.6f}')
        
        # Save model after training
        if self.save_path:
            self.save_model()

    def validation(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_data_loader:
                batch = batch.to(self.device).float()
                input_data, target_data = self.prepare_data(batch)
                output = self.model(input_data)
                loss = self.loss_function(output, target_data)
                val_loss += loss.item()
        return val_loss

    def save_model(self):
        # Save state dict
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def prediction(self):
        self.model.eval()
        outputs_pred = []
        outputs_gt = []
        with torch.no_grad():
            for batch in self.val_data_loader:
                batch = batch.to(self.device).float()
                input_data, target_data = self.prepare_data(batch)
                output = self.model(input_data)
                outputs_pred.append(output.detach().cpu().numpy())
                outputs_gt.append(target_data.detach().cpu().numpy())
        return outputs_pred, outputs_gt
