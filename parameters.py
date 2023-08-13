from enum import Enum, auto

class StemmingOptions(Enum):
    NONE = auto()
    ISRILIGHT = auto()
    ISRI = auto()
    ARABICLIGHT = auto()

class StopWordsOptions(Enum):
    KEEP = auto()
    REMOVE = auto()

def get_marbert_parameters():
    MARBERT = {
        'model_name': "UBC-NLP/MARBERT",
        'preprocessing_args': {
            'raw':True,
            'stemming': StemmingOptions.NONE,
            'stop_words': StopWordsOptions.KEEP
        },
        'training_args': {
            'do_train': True,
            'evaluate_during_training': True,
            'adam_epsilon': 1e-8,
            'learning_rate': 2e-5,
            'warmup_steps': 0,
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
            'num_train_epochs': 1,
            'logging_steps': 50,
            'save_steps': 400,
            'seed': 42,
            'report_to': "",
        },
    }
    return MARBERT


def get_gru_parameters():
    GRU = {
        'model_name': None,
        'preprocessing_args': {
            'raw': False,
            'stemming': StemmingOptions.ISRILIGHT ,
            'stop_words': StopWordsOptions.KEEP
        },
        'training_args': {
            'EPOCHS': 2,
            'BATCH_SIZE': 96,
            'emb_dim': 128,  
            'dropout': 0.15798235844548061,
            'learning_rate': 0.002 , 
            'bi_units': 160, 
            'uni_units': 39,
            'dense_units':42     
        },
    }
    return GRU

def get_logreg_tfidf_parameters():
    LOGREG = {
        'preprocessing_args': {
            'raw':False,
            'stemming': StemmingOptions.ISRILIGHT,
            'stop_words': StopWordsOptions.REMOVE
        },
        'training_args': {
            'C': 0.08858667904100823,
            'max_iter': 57,
            'solver': 'saga',
        }
    }
    return LOGREG
    
def get_logreg_bert_parameters():
    LOGREG = {
        'preprocessing_args': {
            'raw': False,
            'stemming': StemmingOptions.ISRILIGHT,
            'stop_words': StopWordsOptions.REMOVE
        },
        'training_args': {
            'C': 0.08858667904100823,
            'max_iter': 57,
            'solver': 'saga',
        }
    }
    return LOGREG
