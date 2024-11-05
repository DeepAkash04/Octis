import json, pickle, sys, os, time, pandas as pd
if 'E:\\Rashi\\octis\\OCTIS' not in sys.path: sys.path.append('E:\\Rashi\\octis\\OCTIS')

from octis.preprocessing.preprocessing import Preprocessing
import spacy, string
spacy.load('en_core_web_sm')

from spacy.lang.en import stop_words
stop_words=stop_words.STOP_WORDS

def addCustomDataset(datafile, columnName='content', basepath='E:\\Rashi\\octis\\OCTIS\\preprocessed_datasets', dataset_name=f'ds_{int(time.time())}', stop_words=stop_words, custom_stop_words=[]):

    """
    Add custom dataset to octis dashboard

    datafile: path of the data file.
    type: str

    basepath: folder path for preprocessed datasets in octis
    type: str
    default: E:\\Rashi\\octis\\OCTIS\\preprocessed_datasets

    dataset_name: name of the dataset
    type: str
    default: ds_{int(time.time())}

    custom_stop_words: dataset specific stop words
    type: list
    default: []
    
    """

    if type(datafile)==str and datafile[-4:]=='.csv':
        data=pd.read_csv(datafile)
        data.drop_duplicates(inplace=True,ignore_index=True)
        docs=[doc.replace('\n',' ')+' \n' for doc in data[columnName].to_list() if type(doc)==str]

        path=datafile.split('\\')[:-1]
        filename= 'E:\\Rashi\\octis\\helperFiles\\' + dataset_name + '_content.txt'
        
        try:
            os.remove(filename)
        except OSError:
            pass

        with open(filename,'a',encoding="utf-8") as file:
            file.writelines(docs)

        stop_words=list(stop_words)+custom_stop_words
        preprocessor = Preprocessing(punctuation=string.punctuation, stopword_list=stop_words, min_words_docs=50)
        dataset = preprocessor.preprocess_dataset(documents_path=filename)
    
    elif type(datafile)==str and datafile[-4:]=='.txt':
        stop_words=list(stop_words)+custom_stop_words
        preprocessor = Preprocessing(punctuation=string.punctuation, stopword_list=stop_words, min_words_docs=50)
        dataset = preprocessor.preprocess_dataset(documents_path=datafile) 

    elif type(datafile)==str and datafile[-4:]=='.pkl':
        with open(datafile,'rb') as file:
            dataset=pickle.load(file)

    else:
        raise TypeError('Inappropriate argument type for datafile. datafile should be a string and a file path')

    #createDirectory
    if os.path.exists(os.path.join(basepath,dataset_name)):
        pass
    else:
        os.mkdir(os.path.join(basepath,dataset_name))

    #metadataFile
    metadata=dataset.get_metadata()
    with open(os.path.join(basepath, dataset_name,'metadata.json'), 'w', encoding='utf-8') as file:
        json.dump(dataset.get_metadata(), file)
    print(f'saved metadata')

    #corpus
    corpus=dataset.get_corpus()
    pd.DataFrame(corpus).to_csv(os.path.join(basepath, dataset_name,'corpus.tsv'),sep='\t',index=False,header=False)
    print(f'saved corpus')

    #labels
    labels=dataset.get_labels()
    with open(os.path.join(basepath, dataset_name,'labels.txt'), 'w', encoding='utf-8') as file:
        if len(labels)==0: print('labels not provided')
        else: [file.write(f'{label} \n') for label in labels]
    print(f'saved labels')

    #vocab
    vocab=dataset.get_vocabulary()
    with open(os.path.join(basepath, dataset_name,'vocabulary.txt'), 'w', encoding='utf-8') as file:
        [file.write(f'{word} \n') for word in vocab]
    print(f'saved vocab')
    