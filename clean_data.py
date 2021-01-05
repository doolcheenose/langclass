import pandas as pd
import numpy as np
import re
import sys

if len(sys.argv) != 2:
    raise Exception('Incorrect number of arguments. Usage: python3 clean_data.py [output.csv]')

lang_ids = {'english': 190.0, 'spanish': 176.0, 'portuguese': 178.0, 'german': 194.0, 'french': 171.0, 'italian': 170.0}
data_path = './ids_data/ids/raw/ids-data-master/ids.all.csv'
raw_data = pd.read_csv(data_path, usecols=['lg_id', 'data_1'], dtype={'data_1':'string'})

df = pd.DataFrame()

for lg_name, lg_id in lang_ids.items():
    lang_data = raw_data[raw_data.lg_id == lg_id]
    lang_data.columns = ['lg_id', 'word']
    # some rows have multiple words, e.g. 'bat, cat, hat'
    # this will separate these words into their own rows

    for sep in (', ', '; ', '/ ', ' ', '/', '-'): # separators found in dataset
        # found this at stackoverflow.com/questions/12680754/
        lang_data = pd.DataFrame(lang_data.word.str.split(sep).tolist(), 
                                 index=lang_data.lg_id).stack()
        lang_data = lang_data.reset_index()
        lang_data.columns = ['lg_id', 'word_row_ind', 'word']

    stripchars = '[()\[\]?' + chr(191) + '!" ]' # use regex to remove chars ()[]?¿!" from each word
    lang_data.word = lang_data.word.apply(lambda word : re.sub(stripchars, '', word).lower())
    subchars = [(chr(228), 'a' + chr(776)), (chr(246), 'o' + chr(776)), (chr(252), 'u' + chr(776)), (chr(626), chr(241))]
    for (old, new) in subchars:
        lang_data.word = lang_data.word.apply(lambda word : re.sub(old, new, word))

    df = df.append(lang_data)

df = df.reset_index().drop(columns=['index']).astype({'lg_id':'int32'})
df = df.replace({'lg_id': {190: 0, 176: 1, 178: 2, 194: 3, 171: 4, 170: 5}})
conditions = [df.lg_id==i for i in range(6)]
langs = ['english', 'spanish', 'portuguese', 'german', 'french', 'italian']
df['lg_name'] = np.select(conditions, langs)

df = df.drop_duplicates(subset=['word', 'lg_id'])
fil = lambda word: (not isinstance(word, str) or len(word) < 1 or word == 'null')
# apparently 'null' is an actual german word...
# just gonna take it out
df = df.drop(df[df['word'].apply(fil)].index)

df.to_csv(str(sys.argv[1])) 
