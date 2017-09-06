import os.path
import pandas as pd
import numpy as np
from collections import defaultdict
import csv
from tqdm import tqdm

def write_submission(predictions, fp, base_dir='../data/'):
    """Write a submission file from a numpy array containing predictions. MUST
    BE IN THE SAME ORDER in the first dimension as the train_1.csv of pages! It
    just makes sense ok. The predictions will be clipped at 0 and rounded to
    integers.

    Args:
        predictions -- np.array shape (145063, 60) of predictions with the page
        ordering the same as in train_1.csv
        fp - string filepath to save to
    """
    predictions = predictions.round().clip(0).astype(int)
    ids = get_ids_df(base_dir).drop('Page', axis=1).values
    submission = pd.DataFrame(
        data=np.stack([ids.reshape(-1), predictions.reshape(-1)], axis=1),
        columns=['Id', 'Visits']
    )
    submission.to_csv(fp, index=False)

def get_ids_df(data_dir):
    """Reads from file, or creates and saves if necessary, a dataframe with 
    columns:
        Page: each page ordered the same as train_1.csv
        one for each date in the range, ordered by time increasing
    with values as the ids for submission.
    I use it by dropping the Page column and taking the values (see function
    above)

    Args:
        data_dir -- string directory name where the file should be saved
    Returns:
        dataframe described above
    """
    path = os.path.join(data_dir, 'ids_df.f')
    if os.path.isfile(path):
        df = pd.read_feather(path)
    else:
        print('No ids df found. Creating and saving...')
        df = _create_ids_df(data_dir)
        df.to_feather(path)
    return df

def _create_ids_dict(data_dir, daterange):
    """Creates a dictionary of page name to an array of ids time increasing.
    Daterange given as argument to confirm that the ids appear in date
    increasing order in the key.csv."""
    key_dict = defaultdict(list)
    with open(os.path.join(data_dir, 'key_1.csv')) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in tqdm(enumerate(reader)):
            split = row['Page'].split('_')
            ds = split[-1]
            page_name = '_'.join(split[:-1])
            key_dict[page_name].append(row['Id'])
            #Assert the date ordering has been the same
            assert ds == daterange[len(key_dict[page_name])-1]    
    return key_dict

def _create_ids_df(data_dir):
    """Create the ids_df returned by get_ids_df. Date range and train_*.csv
    paths should be changed in stage 2.
    """
    
    #Create a list of the dates used, as strings in the right format
    daterange = list(map(
        lambda x: str(x.date()),
        pd.date_range(start='2017-01-01', end='2017-03-01')
    ))

    print('Creating page to id dictionary...')
    key_dict = _create_ids_dict(data_dir, daterange)
    print('Resorting to same order as train_1.csv...')
    id_df = pd.DataFrame.from_dict(key_dict, orient='index')
    id_df.columns = daterange

    train = pd.read_csv(os.path.join(data_dir, 'train_1.csv'))
    train_df = train.set_index('Page')
    joined = train_df.join(id_df, how='inner')

    #Assert the index is indeed in the same order.
    assert all(train_df.index == joined.index)
    #Here I reset the index since feather doesn't like string indices.
    joined.index = joined.index.rename('Page')
    joined = joined.reset_index()
    return joined[['Page'] + daterange]
    
