
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
        # Handle Multi-Feature Target (B, T, C, F) vs Single Output (B, T, C)
        if target.ndim == 4 and prediction.ndim == 3:
            # Assume we are predicting Feature 0 (Band 0)
            target = target[..., 0]
            
        if self.forecasting_mode == 'one_step':
            loss = self.loss_fn(prediction, target)
        else:
            # Multi-step: compare prediction steps with target steps
            # prediction shape: (B, T, C)
            # target shape: (B, T-init_steps, C)
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
        return outputs_pred, outputs_gt

class AdvancedTrainer(Trainer):
    """
    Trainer with Advanced Strategies: Masked Modeling & Contrastive Loss
    """
    def __init__(self, model, train_data_loader, test_data_loader,
               val_data_loader, loss_fn, optimizer, device, scheduler,
               forecasting_mode, init_steps=10, save_path='', ckpt_path=None,
               mask_ratio=0.15, lambda_recon=0.5, lambda_contrast=0.1):
        super(AdvancedTrainer, self).__init__(model, train_data_loader, test_data_loader,
               val_data_loader, loss_fn, optimizer, device, scheduler,
               forecasting_mode, init_steps, save_path, ckpt_path)
        
        self.mask_ratio = mask_ratio
        self.lambda_recon = lambda_recon
        self.lambda_contrast = lambda_contrast
        
    def nt_xent_loss(self, z_i, z_j, temperature=0.5):
        """
        NT-XEnt Loss (Normalized Temperature-scaled Cross Entropy Loss)
        z_i, z_j: (Batch, Hidden) representations of two augmented views
        """
        batch_size = z_i.shape[0]
        
        # Normalize vectors
        z_i = torch.nn.functional.normalize(z_i, dim=1)
        z_j = torch.nn.functional.normalize(z_j, dim=1)
        
        # Concatenate for similarity matrix
        z = torch.cat([z_i, z_j], dim=0) # (2B, H)
        
        # Similarity matrix: (2B, 2B)
        sim_matrix = torch.matmul(z, z.T) / temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size).to(self.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Positive pairs are (i, batch+i) and (batch+i, i)
        # We need to construct labels
        # For sample k (0 to B-1), positive is k+B
        # For sample k (B to 2B-1), positive is k-B
        
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(self.device)
        
        loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
        return loss

    def random_masking(self, x, mask_ratio):
        """
        Randomly mask x (Batch, T, N) or (Batch, T, N, F)
        Returns: masked_x, mask_matrix (1=masked)
        """
        # Create mask
        mask = torch.rand_like(x) < mask_ratio
        
        # Apply mask (replace with 0 or mean? BERT uses 0 [MASK])
        # Since input is normalized ~N(0,1), 0 is mean.
        x_masked = x.clone()
        x_masked[mask] = 0
        
        return x_masked, mask

    def train(self, num_epochs):
        print(f"Starting Advanced Training (Masking={self.mask_ratio}, Recon={self.lambda_recon}, Contrast={self.lambda_contrast})")
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            recon_loss_acc = 0.0
            contrast_loss_acc = 0.0
            
            for batch in self.train_data_loader:
                batch = batch.to(self.device).float()
                
                # Main Task Data
                input_data, target_data = self.prepare_data(batch)
                
                self.optimizer.zero_grad()
                
                # --- 1. Masked Modeling & Main Prediction ---
                # Mask input_data (Method A: Apply masking to the input we feed the model)
                input_masked, mask = self.random_masking(input_data, self.mask_ratio)
                
                # Forward
                pred, recon, latent = self.model(input_masked, return_aux=True)
                
                # Main Forecasting Loss (on predicting Future)
                # Prediction is based on Masked History. This forces robustness.
                loss_main = self.loss_function(pred, target_data)
                
                # Reconstruction Loss (on masked parts of Input)
                # recon: (Batch, T, N)
                # input_data: (Batch, T, N) (assuming T is init_steps)
                # target is original input_data
                
                # NOTE: input_data from prepare_data can be complex.
                # If multi_step: input_data is (B, 20, C) with repeats?
                # No, prepare_data for multi_step:
                # input_data[:, :init] is history. input_data[:, init:] is repeated last step.
                # We should only reconstruct history?
                # Model slices input to input_len.
                # Recon output matches input_len (10).
                # So we compare recon with input_data[:, :10]
                
                history_target = input_data[:, :self.init_steps]
                if history_target.ndim == 4: history_target = history_target[..., 0]
                
                # Calculate MSE only on masked elements
                # mask is same shape as input_data. Slice it too.
                mask_history = mask[:, :self.init_steps]
                if mask_history.ndim == 4: mask_history = mask_history[..., 0]
                
                # Avoid NaN if no mask
                if mask_history.sum() > 0:
                    loss_recon = torch.nn.functional.mse_loss(
                        recon[mask_history], 
                        history_target[mask_history]
                    )
                else:
                    loss_recon = torch.tensor(0.0).to(self.device)

                # --- 2. Contrastive Loss ---
                # View 1 is input_masked.
                # View 2: Another random mask or jitter?
                # Let's generate View 2
                input_aug2, _ = self.random_masking(input_data, self.mask_ratio)
                _, _, latent2 = self.model(input_aug2, return_aux=True)
                
                # Latent is (Batch, N, Hidden). Pooling to (Batch, Hidden) or (Batch*N, Hidden)?
                # Global context consistency is simpler.
                z1 = latent.mean(dim=1)
                z2 = latent2.mean(dim=1)
                
                loss_contrast = self.nt_xent_loss(z1, z2)
                
                # --- Total Loss ---
                loss = loss_main + self.lambda_recon * loss_recon + self.lambda_contrast * loss_contrast
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss_main.item() # Report main loss
                recon_loss_acc += loss_recon.item()
                contrast_loss_acc += loss_contrast.item()
            
            if self.scheduler:
                self.scheduler.step()

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.model.eval()
                # Use standard validation logic (no masking) for fair comparison
                val_loss = self.validation() 
                self.model.train()
                print(f'Epoch {epoch}, Loss: {train_loss/len(self.train_data_loader):.4f} (Main), {recon_loss_acc/len(self.train_data_loader):.4f} (Recon), {contrast_loss_acc/len(self.train_data_loader):.4f} (Contr) | Val: {val_loss/len(self.val_data_loader):.6f}')
        
        if self.save_path:
            self.save_model()
