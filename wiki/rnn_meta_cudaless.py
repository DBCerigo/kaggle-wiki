import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .utils import clock

class RNN(nn.Module):
    """Class to represent the RNN model with meta features."""
    def __init__(self, loss_func=None, teacher_forcing_ratio=0.5,
            embedding_in=145063, embedding_out=20, num_feats=3):
        """
        Args:
            lost_func -- pytorch loss function. Note only loss functions where
            lower is better are supported currently. (default L1Loss aka MAE)
            teacher_forcing_ratio -- float between 0 and 1. Percentage of time 
            to force the model to train on target data (rather than its own 
            recursive output
            embedding_in -- int number of indices the embedding maps from
            (default 145603 - number of series)
            embedding_out -- int number of dimensions embedding maps to (default
            20)
            num_feats -- int number of self made features (ie age, day of
            week, week of year) (default 3)
        """
        super().__init__()

        self.embedding = torch.nn.Embedding(embedding_in, embedding_out)

        self.hidden_units = 128
        self.n_layers = 2

        self.num_feats = num_feats
        self.embedding_out = embedding_out

        print(1+num_feats+embedding_out)
        self.rnn = nn.GRU(
            input_size = 1+num_feats+embedding_out,
            hidden_size = self.hidden_units,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.out = nn.Linear(self.hidden_units, 1)

        self.loss_func = nn.L1Loss() if loss_func is None else loss_func
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, x, h_state=None):
        r_out, h_state = self.rnn(x, h_state)
        return self.out(r_out), h_state

    def _predict_batch(self, sequence_batch, targets, pred_len):
        output = []
        x=Variable(sequence_batch[:,:,:-1], volatile=True)
        e=Variable(sequence_batch[:,:,-1].long(), volatile=True)
        
        embed = self.embedding(e)
        inp = torch.cat([x, embed], dim=2)

        encoder_out, h_state = self(inp)

        #e is the same for all timesteps so we just pick the last one
        embed_1 = embed[:,-1:,:]

        iv = encoder_out[:,-1:,:]
        output.append(iv)
        for i in range(pred_len-1):
            #We get the time dependent covariates from the 'targets'
            time_dep = t[:,i:i+1,1:-1]
            input_variable = torch.cat([iv, time_dep, embed_1], dim=2)
            encoder_out, h_state = self(input_variable, h_state)
            iv = encoder_out
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
            output = self._predict_batch(sequences, t, pred_len)
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
            targets = Variable(targets, volatile=True)
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
    
                    x=Variable(sequences[:,:,:-1])
                    e=Variable(sequences[:,:,-1].long())
                    #e=Variable(sequences[:,:,-1:], requires_grad=False)
                    y=Variable(targets)

                    embed = self.embedding(e)
                    x = torch.cat([x, embed], dim=2)

                    #run through 'encoder' stage
                    encoder_out, h_state = self(x)

                    #Now 'decoder' stage
                    rand = np.random.rand() 
                    use_teacher_forcing =  rand < self.teacher_forcing_ratio
                    use_teacher_forcing = False
                    #e is the same for all timesteps so we just pick the last
                    #one
                    embed_1 = embed[:,-1:,:]
                    for i in range(y.size()[1]-1):
                        #Get the time dependent features
                        time_dep = y[:,i:i+1,1:-1]
                        if use_teacher_forcing:
                            iv = y[:,i:i+1,:1]
                        else:
                            iv = encoder_out[:,-1:,:]
                        input_variable = torch.cat([iv,time_dep,embed_1], dim=2)
                        encoder_out, h_state = self(input_variable, h_state)
                        loss += self.loss_func(
                            encoder_out, y[:,i+1:i+2,:1]
                        )

                    optimizer.zero_grad()                   
                    loss.backward()
                    running_total+=loss.data[0]
                    if step>0 and step % 5 == 0:
                        #Print running average of loss for thie epoch
                        running_avg = running_total / (step*y.size()[1])
                        print('Running average loss: %f' % running_avg,end='\r')
                    optimizer.step()
                    step += 1
                print('')
                if valloader is not None:
                    average_loss = self.validate(valloader)
                    print('VALIDATION LOSS: %f' % float(average_loss))
                    if save_best_path is not None and average_loss<best_val_loss:
                        best_val_loss = average_loss
                        torch.save(self, save_best_path)
