import datetime
import numpy as np
import matplotlib as mpl
import os
if not "DISPLAY" in os.environ:
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import sys
sys.path.insert(0,'..')
sys.path.append('../architectures/')
import importlib
from importlib.machinery import SourceFileLoader
import algorithms.EarlyStopping as ES
from datetime import datetime


# ---
# ====================== Training functions ====================== #
# ---
class APM_algorithm():
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt['batch_size']
        self.epochs = opt['epochs']
        self.snapshot = self.opt['snapshot']
        self.console_print = self.opt['console_print']
        
        self.lr_schedule = opt['lr_schedule']
        self.init_lr_schedule = opt['lr_schedule']
        
        self.current_epoch = None
        self.model = None
        self.optimiser = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(' *- Chosen device: ', self.device)
        
        print(' *- Random seed: ', opt['random_seed'])
        self.random_seed = opt['random_seed']
        torch.manual_seed(opt['random_seed'])
        np.random.seed(opt['random_seed'])
        if self.device == 'cuda': torch.cuda.manual_seed(opt['random_seed'])
        self.save_path = self.opt['exp_dir'] + '/' + self.opt['filename']
        self.model_path = self.save_path + '_model.pt'
    
        self.best_model= {
                'model': self.model,
                'epoch': self.current_epoch,
                'train_loss': None, 
                'valid_loss': None
                }

    
    def count_parameters(self):
        """Counts the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    
    def descale_coords(self, x):
        """
        Descales the coordinates from [0, 1] interval back to the original
        image size.
        """
        rescaled = x.cpu().numpy() * (self.data_max - self.data_min) + self.data_min
        rounded_coords = np.around(rescaled).astype(int)
        
        # Filter out of the range coordinates because MSE can be out
        cropped_rounded_coords = np.maximum(self.data_min, np.minimum(rounded_coords, self.data_max))
        assert((cropped_rounded_coords >= self.data_min).all())
        assert((cropped_rounded_coords <= self.data_max).all())
        return cropped_rounded_coords.astype(int)
    
    
    def plot_snapshot_loss(self):
        """Plots epochs vs model loss for a given range of epochs."""
        plt_data = np.stack(self.epoch_losses)
        for i in range(len(self.action_labels)):
            plt.subplot(len(self.action_labels),1,i+1)
            plt.plot(np.arange(self.snapshot)+(self.current_epoch//self.snapshot)*self.snapshot,
                     plt_data[self.current_epoch-self.snapshot+1:self.current_epoch+1, i], 
                     label=self.action_labels[i])
            plt.ylabel(self.action_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_SnapshotLosses_{0}'.format(self.current_epoch))
        plt.clf()
        plt.close()
    
    
    def plot_model_loss(self, input_idx, output_idx, model_loss_idx):
        """Plots epochs vs model loss."""
        # All losses
        plt_data = np.stack(self.epoch_losses)
        for i in range(len(self.action_labels)):
            plt.subplot(len(self.action_labels),1,i+1)
            plt.plot(np.arange(self.current_epoch+1),
                     plt_data[:, i], 
                     label=self.action_labels[i])
            plt.ylabel(self.action_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_Losses')
        plt.clf()
        plt.close()
        
        # Losses on the input coordinates
        fig, ax = plt.subplots()
        ax.plot(plt_data[:, input_idx[0]], 'g-', linewidth=2, label='inputX loss')
        ax.plot(plt_data[:, input_idx[1]], 'r-', linewidth=2, label='inputY loss')
        ax.plot()
        ax.legend()
        ax.set_xlim(0, self.epochs)
        ax.set(xlabel='# epochs', ylabel='loss', title='Input Coordinate loss')
        plt.savefig(self.save_path + '_InputCoordLoss')
        plt.close()
        
        # Losses on the output coordinates
        fig, ax = plt.subplots()
        ax.plot(plt_data[:, output_idx[0]], 'g-', linewidth=2, label='outputX loss')
        ax.plot(plt_data[:, output_idx[1]], 'r-', linewidth=2, label='outputY loss')
        ax.plot()
        ax.legend()
        ax.set_xlim(0, self.epochs)
        ax.set(xlabel='# epochs', ylabel='loss', title='Output Coordinate loss')
        plt.savefig(self.save_path + '_OutputCoordLoss')
        plt.close()
        
        # Total model loss
        fig2, ax2 = plt.subplots()
        ax2.plot(plt_data[:, model_loss_idx], 'go-', linewidth=3, label='Model loss')
        ax2.plot()
        ax2.set_xlim(0, self.epochs)
        ax2.set(xlabel='# epochs', ylabel='loss', title='Model loss')
        plt.savefig(self.save_path + '_Loss')
        plt.close()
    
    
    
    def plot_test_images(self, valid_dataset):
        """Plots sthe APN predictions on a subset of test set."""
        self.model.eval()
        assert(not self.model.training)
        
        batch_size = 5
        valid_batch = torch.utils.data.Subset(valid_dataset, np.arange(0, 100, step=3))
        valid_dataloader = torch.utils.data.DataLoader(
                valid_batch, batch_size, drop_last=True)
        
        for batch_idx, (latent1, latent2, coords) in enumerate(valid_dataloader):
            latent1 = latent1.to(self.device)
            latent2 = latent2.to(self.device)
            coords = coords.float().to(self.device)
            
            # APNet loss
            pred_coords = self.model(latent1, latent2)                    
            self.plot_prediction(latent1, latent2, pred_coords, coords, 
                                 split='AfterTraining' + str(batch_idx))

    
    def plot_prediction(self, *args):
        """
        Plots the APN predictions on the given (no-)action pair.
        
        Defined in each subclass.
        """
        pass
        
        
    def plot_learning_curve(self):
        """Plots train and validation learning curves of the APN training."""
        train_losses_np = np.stack(self.epoch_losses)
        valid_losses_np = np.stack(self.valid_losses)
        assert(len(valid_losses_np) == len(train_losses_np))
        
        for i in range(len(self.action_labels)):
            plt.subplot(len(self.action_labels),1,i+1)
            plt.plot(train_losses_np[:, i], 'g-', linewidth=2, 
                     label='Train ' + self.action_labels[i])
            plt.plot(valid_losses_np[:, i], 'b--', linewidth=2, 
                     label='Valid ' + self.action_labels[i])
            plt.ylabel(self.action_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_chpntValidTrainLoss')
        plt.clf()
        plt.close()
        
        
    def plot_epoch_time(self):
        """Plots elapsed time for each epoch"""
        epoch_times = list(map(lambda x: x.total_seconds(), self.epoch_times))
        mean_time = np.mean(epoch_times)
        plt.figure(1)
        plt.clf()
        plt.plot(np.arange(len(epoch_times)), epoch_times, linewidth=2, 
                 label='observed')
        plt.plot([0, len(epoch_times)], [mean_time, mean_time], 'g--', 
                 alpha=0.7, linewidth=1, label='average')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('elapsed time')
        plt.savefig(self.save_path + '_chpntTime')
        plt.clf()
        plt.close()
        

    def compute_loss(self, pred_coords, coords):
        """Computes the loss on the training batch given the criterion."""
        batch_loss = nn.MSELoss(reduction='none')(pred_coords, coords) # (batch, n_action_coords)
        per_feat_loss = torch.mean(batch_loss, dim=0) # n_action_coords
        the_loss = torch.sum(per_feat_loss)
        return the_loss, per_feat_loss
    
    
    def compute_test_loss(self, valid_dataset):
        """Computes the loss on a test dataset."""
        self.model.eval()
        assert(not self.model.training)
        
        batch_size = min(len(valid_dataset), self.batch_size)
        valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, batch_size, drop_last=True)
        
        losses = np.zeros(7)
        for batch_idx, (latent1, latent2, coords) in enumerate(valid_dataloader):
            latent1 = latent1.to(self.device)
            latent2 = latent2.to(self.device)
            coords = coords.float().to(self.device)
            
            # APNet loss
            pred_coords = self.model(latent1, latent2)                
            (the_loss, per_actionFeature_Loss) = self.compute_loss(pred_coords, coords) 
            losses += self.format_loss([the_loss, *per_actionFeature_Loss]) 
            
        if (self.current_epoch + 1) % self.snapshot == 0:    
            self.plot_prediction(latent1, latent2, pred_coords, coords, split='test')
            
        n_valid = len(valid_dataloader)
        return losses / n_valid
    
    
    def format_loss(self, losses_list):
        """Rounds the loss and returns an np array for logging."""
        reformatted = list(map(lambda x: round(x.item(), 2), losses_list))
        reformatted.append(int(self.current_epoch))
        return np.array(reformatted)
    
    
    def load_vae(self):
        """Loads a pretrained VAE model for encoding the states."""
        vae_name = self.opt['vae_name']
        path_to_pretrained = 'models/{0}/vae_model.pt'.format(vae_name)
        vae_config_file = os.path.join('configs', vae_name + '.py')
        vae_config = SourceFileLoader(vae_name, vae_config_file).load_module().config 

        vae_opt = vae_config['vae_opt']
        vae_opt['device'] = self.device
        vae_opt['vae_load_checkpoint'] = False
        
        vae_module = importlib.import_module("architectures.{0}".format(vae_opt['model']))
        print(' *- Imported module: ', vae_module)
        
        # Initialise the model
        try:
            class_ = getattr(vae_module, vae_opt['model'])
            vae_instance = class_(vae_opt).to(self.device)
            print(' *- Loaded {0} from {1}.'.format(class_, vae_name))
        except: 
            raise NotImplementedError(
                    'Model {0} not recognized'.format(vae_opt['model']))
        
        # Load the weights
        if vae_opt['vae_load_checkpoint']:
            checkpoint = torch.load(path_to_pretrained, map_location=self.device)
            vae_instance.load_state_dict(checkpoint['model_state_dict'])
            print(' *- Loaded checkpoint.')
        else:
            vae_instance.load_state_dict(torch.load(path_to_pretrained, map_location=self.device))
        vae_instance.eval()
        assert(not vae_instance.training)
        self.vae = vae_instance
    
    
    def init_model(self):
        """Initialises the APN model."""
        model = importlib.import_module("architectures.{0}".format(self.opt['model_module']))
        print(' *- Imported module: ', model)
        try:
            class_ = getattr(model, self.opt['model_class'])
            instance = class_(self.opt).to(self.device)
            return instance
        except: 
            raise NotImplementedError(
                    'Model {0} not recognized'.format(self.opt['model_module']))
        
        
    def init_optimiser(self):
        """Initialises the optimiser."""
        print(self.model.parameters())
        if self.opt['optim_type'] == 'Adam':
            print(' *- Initialised Adam optimiser.')
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt['optim_type'] == 'RMSprop':
            print(' *- Initialised RMSprop optimiser.')
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
        else: 
            raise NotImplementedError(
                    'Optimiser {0} not recognized'.format(self.opt['optim_type']))
    
    
    def update_learning_rate(self, optimiser):
        """Annealing schedule for learning rates."""
        if self.current_epoch == self.lr_update_epoch:
            for param_group in optimiser.param_groups:
                self.lr = self.new_lr
                param_group['lr'] = self.lr
                print(' *- Learning rate updated - new value:', self.lr)
                try:
                    self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
                except:
                    print(' *- Reached the end of the update schedule.')
                print(' *- Remaining lr schedule:', self.lr_schedule)
                
    
    def monitor_training(self):
        """
        Monitoring functions.
        """    
        pass
    
    
    def train(self, train_dataset, test_dataset, num_workers=0, chpnt_path=''):
        """Trains an APM model with given hyperparameters."""
        dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=num_workers, drop_last=False) 
        n_data = len(train_dataset)
        self.data_min = train_dataset.min.reshape(1, -1)
        self.data_max = train_dataset.max.reshape(1, -1)
        print(' *- Threshold shapes: min {0}, max {1}'.format(self.data_min.shape, self.data_max.shape))

        print(('\nPrinting model specifications...\n' +
               ' *- Path to the model: {0}\n' +
               ' *- Training dataset: {1}\n' +
               ' *- Number of training samples: {2}\n' +
               ' *- Number of epochs: {3}\n' +
               ' *- Batch size: {4}\n'
               ).format(self.model_path, train_dataset.dataset_name, n_data,
                   self.epochs, self.batch_size))

        if chpnt_path:
            # Pick up the last epochs specs
            self.load_checkpoint(chpnt_path)

        else:
            # Initialise the model
            self.model = self.init_model()
            self.start_epoch, self.lr = self.lr_schedule.pop(0)
            try:
                self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
            except:
                self.lr_update_epoch, self.new_lr = self.start_epoch - 1, self.lr
            self.optimiser = self.init_optimiser()
            self.training_losses = []
            self.valid_losses = []
            self.epoch_losses = []
            self.epoch_times = []
            self.training_time = 0
            print((' *- Learning rate: {0}\n' +
                   ' *- Next lr update at {1} to the value {2}\n' +
                   ' *- Remaining lr schedule: {3}'
                   ).format(self.lr, self.lr_update_epoch, self.new_lr,
                   self.lr_schedule))

        self.load_vae()
        es = ES.EarlyStopping(patience=20)
        num_parameters = self.count_parameters()
        self.opt['num_parameters'] = num_parameters
        print(' *- Model parameter/training samples: {0}'.format(
                num_parameters/len(train_dataset)))
        print(' *- Model parameters: {0}'.format(num_parameters))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                spacing = 1
                print('{0:>2}{1}\n\t of dimension {2}'.format('', name, spacing),
                      list(param.shape))

        print('\nStarting to train the model...\n' )
        training_start = datetime.now()
        for self.current_epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self.update_learning_rate(self.optimiser)
            epoch_loss = np.zeros(7)
            epoch_start = datetime.now()

            for batch_idx, (latent1, latent2, coords) in enumerate(dataloader):
                latent1 = latent1.to(self.device)
                latent2 = latent2.to(self.device)
                coords = coords.float().to(self.device)

                # APNet loss
                pred_coords = self.model(latent1, latent2)
                (the_loss, per_actionFeature_Loss) = self.compute_loss(pred_coords, coords)
                
                # Optimise the model
                self.optimiser.zero_grad()
                the_loss.backward()
                self.optimiser.step()

                # Monitoring the learning
                epoch_loss += self.format_loss([the_loss, *per_actionFeature_Loss])
            
            epoch_end = datetime.now()
            epoch_time = epoch_end - epoch_start
            self.epoch_times.append(epoch_time)
            self.training_time = training_start - epoch_end
            
            epoch_loss /= len(dataloader)
            epoch_loss[-1] = int(self.current_epoch)
            self.epoch_losses.append(epoch_loss)
            
            valid_loss = self.compute_test_loss(test_dataset)
            valid_loss[-1] = int(self.current_epoch)
            self.valid_losses.append(valid_loss)
            
            # Print current loss values every epoch
            if (self.current_epoch + 1) % self.console_print == 0:
                print('Epoch {0}: [{1}]'.format(self.current_epoch, epoch_time))
                for name, train_value, valid_value in list(zip(self.action_labels, epoch_loss[:-1], valid_loss[:-1])):
                    print('   Train/Valid {0}: {1:.5f}/{2:.5f}'.format(name, train_value, valid_value))
            
            # Update the best model
            try:
                if es.keep_best(valid_loss[0]):
                    self.best_model= {
                            'model': self.model.state_dict().copy(),
                            'epoch': self.current_epoch,
                            'train_loss': epoch_loss[0],
                            'valid_loss': valid_loss[0]
                        }
                    print(' *- New best model at epoch ', self.current_epoch)
            except AssertionError:
                break

            # Update the checkpoint only if there was no early stopping
            self.save_checkpoint(epoch_loss[0])
            self.monitor_training()
            
            # Print validation results when specified
            if (self.current_epoch + 1) % self.snapshot == 0:

                # Plot APN predictions
                self.plot_prediction(latent1, latent2, pred_coords, coords)
                self.model.eval()

                # Plot training and validation loss
                self.save_checkpoint(epoch_loss[0], keep=True)

                # Write logs
                self.save_logs(train_dataset, test_dataset)
                self.plot_snapshot_loss()
            

        print('Training completed.')
        training_end = datetime.now()
        self.training_time = training_end - training_start
        self.monitor_training()
        self.model.eval()
        
        # Save the model
        self.save_checkpoint(epoch_loss[0], keep=True)
        torch.save(self.best_model['model'], self.model_path)
        torch.save(self.model.state_dict(), self.save_path + '_lastModel.pt')
        self.save_logs(train_dataset, test_dataset)

        # Plot predetermined test images for a fair comparisson among models
        self.plot_test_images(test_dataset)
        
        
    def score_model(self, *args):
        """
        Scores a trained model on the test set.
        Defined in each subclass.
        """ 
        pass

    
    def save_logs(self, train_dataset, test_dataset):
        """Save training logs."""
        log_filename = self.save_path + '_logs.txt'
        valid_losses = np.stack(self.valid_losses)
        epoch_losses = np.stack(self.epoch_losses)
        epoch_times = list(map(lambda t: t.total_seconds(), self.epoch_times))
            
        with open(log_filename, 'w') as f:
            f.write('Model {0}\n\n'.format(self.opt['filename']))
            f.write( str(self.opt) )
            f.writelines(['\n\n', 
                    '*- Model path: {0}\n'.format(self.model_path),
                    '*- Training dataset: {0}\n'.format(train_dataset.dataset_name),
                    '*- Number of training examples: {0}\n'.format(len(train_dataset)),
                    '*- Model parameters/Training examples ratio: {0}\n'.format(
                            self.opt['num_parameters']/len(train_dataset)),
                    '*- Number of testing examples: {0}\n'.format(len(test_dataset)),
                    '*- Learning rate schedule: {0}\n'.format(self.init_lr_schedule),
                    '*- Average epoch time {0}\n'.format(round(np.mean(epoch_times), 5)),
                    '*- Training time: {0}\n'.format(round(self.training_time.total_seconds(), 5)),
                    '*- Best model performance at epoch {0}\n'.format(self.best_model['epoch']),
                    ])
            for label_idx, label in enumerate(self.action_labels):
                f.write(f'*- Train/validation {label}\n')
                f.writelines(list(map(
                        lambda t, v, e, time: '{0:>3}Epoch {3:.0f} [{4}] {1:.3f}/{2:.3f}\n'.format('', t, v, e, time), 
                        epoch_losses[:, label_idx], valid_losses[:, label_idx], epoch_losses[:, -1], self.epoch_times)))
                
            f.write('*- Epoch times\n')
            f.writelines(list(map(
                    lambda e, time: '{0:>3}Epoch {1:.0f} {2:.3f} {3}\n'.format('', e[0], e[1], time), 
                    enumerate(epoch_times), self.epoch_times)))
            
        print(' *- Model saved.\n')
    
    
    def save_checkpoint(self, epoch_ml, keep=False):
        """Saves a checkpoint during the training."""
        if keep:
            path = self.save_path + '_checkpoint{0}.pth'.format(self.current_epoch)
            checkpoint_type = 'epoch'
        else:
            path = self.save_path + '_lastCheckpoint.pth'
            checkpoint_type = 'last'
        training_dict = {
                'last_epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimiser_state_dict': self.optimiser.state_dict(),
                'last_epoch_loss': epoch_ml,
                'valid_losses': self.valid_losses,
                'epoch_losses': self.epoch_losses,
                'epoch_times': self.epoch_times,
                'training_time': self.training_time,
                'snapshot': self.snapshot,
                'console_print': self.console_print,
                'current_lr': self.lr,
                'lr_update_epoch': self.lr_update_epoch, 
                'new_lr': self.new_lr, 
                'lr_schedule': self.lr_schedule
                }
        torch.save({**training_dict, **self.opt}, path)
        print(' *- Saved {1} checkpoint {0}.'.format(self.current_epoch, checkpoint_type))
    
        
    def load_checkpoint(self, path, evalm=False, time=True):
        """
        Loads a checkpoint and initialises the models to continue training.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model = self.init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.lr = checkpoint['current_lr']
        self.lr_update_epoch = checkpoint['lr_update_epoch']
        self.new_lr = checkpoint['new_lr']
        self.lr_schedule = checkpoint['lr_schedule']
        self.optimiser= self.init_optimiser()
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
                
        self.start_epoch = checkpoint['last_epoch'] + 1
        self.current_epoch = self.start_epoch - 1
        self.snapshot = checkpoint['snapshot']

        self.valid_losses = checkpoint['valid_losses']
        self.epoch_losses = checkpoint['epoch_losses']

        self.epoch_times = 0
        self.training_time = 0

        if time or 'epoch_times' in checkpoint.keys():
        	self.epoch_times = checkpoint['epoch_times']
        	self.training_time = checkpoint['training_time']
        
        self.snapshot = checkpoint['snapshot']
        self.console_print = checkpoint['console_print']
         
        print(('\nCheckpoint loaded.\n' + 
               ' *- Last epoch {0} with loss {1}.\n' 
               ).format(checkpoint['last_epoch'], 
               checkpoint['last_epoch_loss']))
        print(' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                self.lr, self.lr_update_epoch, self.new_lr)
              )
        if evalm == False:
            self.model.train()
        else: 
            self.model.eval()
    
    def load_best_model_pkl(self, path):
        """Loads the best performing model."""
        best_model_dict = torch.load(path, map_location=self.device)
        best_model_state_dict = best_model_dict['model']
        best_model_trained_params = best_model_dict['trained_params']
        
        self.model = self.init_model(trained_params=best_model_trained_params)
        self.model.load_state_dict(best_model_state_dict)
        self.model.eval()
    
        
