import os.path
import numpy as np
import pandas as pd
import feather

def get_pages_df(data_dir):
	pages_fp = data_dir+'pages_df.f'
	if os.path.isfile(pages_fp):
		pages_df = pd.read_feather(pages_fp)
	else:
		train_df = pd.read_csv(data_dir+'train_1.csv')
		pages_df = train_df.Page.to_frame().reset_index()
		pages_df.columns = ["id", "page"]
		pages_df.to_feather(pages_fp)
	return pages_df

def get_pred_df_template(data_dir):
	template_fp = data_dir+'pred_df_template.f'
	if os.path.isfile(template_fp):
		template = pd.read_feather(template_fp)
	else:
		train_df = pd.read_csv(data_dir+'train_1.csv')
		dates = train_df.drop('Page', axis=1).transpose().index.values
        #Multiple index columns - page IDs, then time series + predictions
		template = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                list(range(len(train_df))),
                ['y', 'y_org', 'yhat', 'train']
            )
        )
        template['ds'] = dates
        template['train'] = [1]*490 + [0]*60
        template.to_feather(template_fp)
    return template

def combine_prediction_data(outputs, targets, sequences):
    predictions_end = outputs.cpu().data.numpy().squeeze()
    truth_end = targets.cpu().numpy().squeeze()
    truth_start = sequences.cpu().numpy().squeeze()

    truth = np.concatenate((truth_start, truth_end), axis=1) ; truth.shape
    predictions = np.concatenate((truth_start, predictions_end), axis=1)
    
    return truth, predictions

def create_prediction_df(outputs, targets, sequences, scaler, pred_df_template):
	
	truth, predictions = combine_prediction_data(outputs, targets, sequences)
    truth = scaler.inverse_transform(truth.T).T
    predictions = scaler.inverse_transform(predictions.T).T

    pred_df = pred_df_template.copy(True)
	for i, (p, t) in enumerate(zip(predictions, truth)):
		pred_df[i, 'y'] = pred_df[i, 'y_org'] = t
        pred_df[i, 'yhat'] = p

    return pred_df

def create_page_df():
	pass
