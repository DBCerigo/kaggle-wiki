import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import StandardScaler
from .utils import clock

def scale_values(X):
    """Scales and reshapes a numpy array of time series, and returns the scaler
    for later unscaling.

    Args:
        X -- numpy array of shape (145063, 550) holding time series data
    Returns:
        X -- numpy array of shape (145063, 550, 1) with the time series data
        rescaled around mean 0
        sc -- sklearn scaler with .inverse_transform method to convert back to
        original scaling

    """

    sc = StandardScaler()
    X = sc.fit_transform(X.T).T
    assert(np.isclose(np.mean(X[0]),0))
    X = X.reshape(X.shape + (1,))
    return X, sc 

class RNN(nn.Module):
    """Class to represent the RNN model.
	
    Todo:
        - implement ideas from (https://arxiv.org/pdf/1704.04110.pdf):  
            - outputting mean and variance and maximising log likelihood of 
            negative binomial distribution
            - including extra features specifically day of week, week/month of
            year, and 
            - an embedding to capture groupings of like pages
    """
    def __init__(self, loss_func=None, teacher_forcing_ratio=0.5):
        """
        Args:
            lost_func -- pytorch loss function. Note only loss functions where
            lower is better are supported currently. (default L1Loss aka MAE)
            teacher_forcing_ratio -- float between 0 and 1. Percentage of time 
            to force the model to train on target data (rather than its own 
            recursive output
        """
        super().__init__()
         
        self.hidden_units = 128
        self.n_layers = 2
        
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=self.hidden_units,
            num_layers=self.n_layers, #number of RNN layers
            batch_first=True, #batch dimension is first
            dropout=0.2
        )

        #I can change the below to two softplus outputs for
        #mean and variance in the paper version (see notes below)
        self.out = nn.Linear(self.hidden_units, 1)

        self.loss_func = nn.L1Loss() if loss_func is None else loss_func
        self.teacher_forcing_ratio = 0.5
        
    def forward(self, x, h_state):
        # dimensions:
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        return self.out(r_out), h_state
    
    def init_hidden(self, batch_size):
        hidden = Variable(
			torch.zeros(self.n_layers, batch_size, self.hidden_units
		)).cuda()
        return hidden

    def _predict_batch(self, sequence_batch, pred_len):
        output = []
        h_state = self.init_hidden(sequence_batch.size()[0])
        x=Variable(sequence_batch, volatile=True).cuda()
        encoder_out, h_state = self(x, h_state)

        input_variable = encoder_out[:,-1:,:]
        output.append(input_variable)
        for i in range(pred_len-1):
            encoder_out, h_state = self(input_variable, h_state)
            input_variable = encoder_out
            output.append(encoder_out)
        
        return torch.cat(output, dim=1)

    def predict(self, dataloader):
        """Given a data loader, predict the next steps in the time series and 
        return predictions and the whole time series for analysis. 

        Args:
            dataloader -- a pytorch dataloader representing batches of the data 
            to predict on
        Returns:
            3-tuple: np.array(RNN output), np.array(sequences fed in), 
                np.array(target values)
        """
        all_output = []
        all_targets = []
        all_sequences = []
        for sequences, t in dataloader:
            pred_len = t.size()[1]
            output = self._predict_batch(sequences, pred_len)
            all_output.append(output)
            all_targets.append(t)
            all_sequences.append(sequences)
        cat = lambda x: torch.cat(x, dim=0)
        return cat(all_output), cat(all_targets), cat(all_sequences)

    def validate(self, valloader):
        """Predict on a dataloader and return the average loss against the 
        targets.
        
        Args:
            valloader -- pytorch DataLoader representing validation set
        Return:
            float average loss 
        """
        loss = 0
        steps = 0
        for sequences, targets in valloader:
            pred_len = targets.size()[1]
            #The flag volatile=True is essential to stop pytorch storing data 
            #for backprop and using all GPU memory
            targets = Variable(targets, volatile=True).cuda()
            output = self._predict_batch(sequences, pred_len)
            loss += self.loss_func(output, targets)
            steps+=1
        average_loss = loss.data[0]/steps
        return float(average_loss)

    def fit(self, trainloader, valloader, optimizer, num_epochs=1, 
            save_best_path=None):
        """Fit the model to data. Prints a running average loss across the
        epoch, then assesses model performance against a validation set. If a 
        filepath is given, saves the best (val set) performing model there.

        Args:
            trainloader -- pytorch DataLoader with training data
            valloader -- pytorch DataLoader with validation data
            optimizer -- pytorch optimizer 
            num_epochs -- int number of epochs to train (default 1)
            save_best_path -- string if given, saves the best (val set) 
            performing model here
        """
        #Setting batch size here to 1 since we'll just be using it to keep
        #track of the size of the batch before
        batch_size = 1 
        best_val_loss = np.inf
        for epoch in range(num_epochs):
            with clock():
                print('\nEPOCH %d' % (epoch+1))
                running_total=0
                step=0
                for sequences, targets in trainloader: 
                    #Restart the loader if the last batch is too small
                    if batch_size and sequences.size(0)<batch_size:
                        continue
                    batch_size = sequences.size(0)

                    loss = 0
    
                    h_state = self.init_hidden(batch_size)
                    x=Variable(sequences).cuda()
                    y=Variable(targets).cuda()

                    #run through 'encoder' stage
                    encoder_out, h_state = self(x, h_state) 

                    #Now 'decoder' stage
                    rand = np.random.rand() 
                    use_teacher_forcing =  rand < self.teacher_forcing_ratio
                    for i in range(y.size()[1]-1):
                        if use_teacher_forcing:
                            input_variable = y[:,i:i+1,:] 
                        else:
                            input_variable = encoder_out[:,-1:,:]
                        encoder_out, h_state = self(input_variable, h_state)
                        loss += self.loss_func(encoder_out, y[:,i+1:i+2,:])

                    optimizer.zero_grad()                   
                    loss.backward()
                    running_total+=loss.data[0]
                    if step>0 and step % 5 == 0:
                        #Print running average of loss for thie epoch
                        running_avg = running_total / (step*y.size()[1])
                        print('Running average loss: %f' % running_avg,end='\r')
                    optimizer.step()
                    step += 1
                average_loss = self.validate(valloader)
                print('')
                print('VALIDATION LOSS: %f' % float(average_loss))
                if save_best_path is not None and average_loss<best_val_loss:
                    best_val_loss = average_loss
                    torch.save(self, save_best_path)
