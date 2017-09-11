import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .utils import clock


def get_page_groups(train_df):
    pages_dict = {}
    ids = []
    for page in train_df.Page:
        w = 'wikipedia.org'
        x = page.find(w)
        name = page[:x+len(w)]
        if name in pages_dict:
            ids.append(pages_dict[name])
        else:
            idnum = len(pages_dict)
            ids.append(idnum)
            pages_dict[name] = idnum
    return ids

class RNN(nn.Module):
    """Class to represent the RNN model with meta features."""
    def __init__(self, loss_func=None, teacher_forcing_ratio=0.5,
            num_feats=3, dropout=0.2):
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

        self.hidden_units = 128
        self.n_layers = 2

        self.num_feats = num_feats

        self.rnn = nn.GRU(
            input_size = 1+num_feats,
            hidden_size = self.hidden_units,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=dropout
        ).cuda()
        
        self.out = nn.Linear(self.hidden_units, 1).cuda()

        self.loss_func = nn.L1Loss() if loss_func is None else loss_func
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, x, h_state=None):
        r_out, h_state = self.rnn(x, h_state)
        return self.out(r_out), h_state

    def _predict_batch(self, batch, pred_len):
        output = []
        series, timefts = batch[:2]
        t_timefts = batch[4]

        series = Variable(series, volatile=True).cuda()
        timefts = Variable(timefts, volatile=True).cuda()
        t_timefts = Variable(t_timefts, volatile=True).cuda()

        inp = torch.cat([series, timefts], dim=-1)

        encoder_out, h_state = self(inp)

        iv = encoder_out[:,-1:,:]
        output.append(iv)
        for i in range(pred_len-1):
            time_dep = t_timefts[:,i:i+1,:]
            input_variable = torch.cat([iv, time_dep], dim=2)
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
        for batch in valloader:
            output = self._predict_batch(batch, pred_len)
            all_output.append(output)
        return torch.cat(all_output, dim=0)

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
        for batch in valloader:
            t_series = batch[3]
            pred_len = t_series.size()[1]
            #The flag volatile=True is essential to stop pytorch storing data 
            #for backprop and using all GPU memory
            t_series = Variable(t_series, volatile=True).cuda()
            output = self._predict_batch(batch, pred_len)
            loss += self.loss_func(output, t_series)
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
        best_val_loss = np.inf
        c = clock()
        for epoch in range(num_epochs):
            c.__enter__()
            print('\nEPOCH %d' % (epoch+1))
            running_total=0
            step=0
            for batch in trainloader: 
                series, timefts = batch[:2]
                t_series, t_timefts = batch[3:-1]
                
                loss = 0

                series = Variable(series).cuda()
                timefts = Variable(timefts).cuda()
                t_timefts = Variable(t_timefts).cuda()

                x = torch.cat([series, timefts], dim=-1)

                y=Variable(t_series).cuda()

                #run through 'encoder' stage
                encoder_out, h_state = self(x)

                #Now 'decoder' stage
                rand = np.random.rand() 
                use_teacher_forcing =  rand < self.teacher_forcing_ratio
                #e is the same for all timesteps so we just pick the last
                #one
                for i in range(y.size()[1]-1):
                    #Get the time dependent features
                    time_dep = t_timefts[:,i:i+1,:]
                    if use_teacher_forcing:
                        iv = y[:,i:i+1,:]
                    else:
                        iv = encoder_out[:,-1:,:]
                    input_variable = torch.cat([iv,time_dep], dim=2)
                    encoder_out, h_state = self(input_variable, h_state)
                    loss += self.loss_func(
                        encoder_out, y[:,i+1:i+2,:]
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
                    torch.save(self.state_dict(), save_best_path)
            c.__exit__(None,None,None)
