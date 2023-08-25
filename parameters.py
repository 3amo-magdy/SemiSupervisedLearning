from enum import Enum, auto

class StemmingOptions(Enum):
    NONE = 0
    ISRILIGHT = 1
    ISRI = 2
    ARABICLIGHT = 3

class StopWordsOptions(Enum):
    KEEP = 0
    REMOVE = 1

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
            'report_to': None,
        },
    }
    return MARBERT

# do not delete this comment

# def get_gru_parameters():
#     GRU = {
#         'model_name': None,
#         'preprocessing_args': {
#             'raw': False,
#             'stemming': StemmingOptions.ISRILIGHT ,
#             'stop_words': StopWordsOptions.KEEP
#         },
#         'training_args': {
#             'EPOCHS': 2,
#             'BATCH_SIZE': 96,
#             'emb_dim': 128,  
#             'dropout': 0.15798235844548061,
#             'learning_rate': 0.002 , 
#             'bi_units': 160, 
#             'uni_units': 39,
#             'dense_units':42     
#         },
#     }
#     return GRU

# FIRST 
# =====================
# 'EPOCHS': 38,
# 'BATCH_SIZE': 96,
# 'emb_dim': 320,  
# 'dropout': 0.6164959,
# 'learning_rate': 0.003007, 
# 'bi_bool': True,
# 'uni_layers': 1,
# 'bi_units': 72, 
# 'uni_units': 104,
# 'final_layer_units': 11,
# 'dense_units':60    

#  SECOND 
# =====================
# 'EPOCHS': 20,
# 'BATCH_SIZE': 96,
# 'emb_dim': 460,  
# 'dropout':  0.1579824,
# 'learning_rate':0.01, 
# 'bi_bool': True,
# 'uni_layers': 0,
# 'bi_units': 160, 
# 'uni_units': 10,
# 'final_layer_units': 39,
# 'dense_units':42    

#  THIRD 
# =====================
# 'EPOCHS': 16,
# 'BATCH_SIZE': 96,
# 'emb_dim': 137,  
# 'dropout':  0.2767830,
# 'learning_rate':0.01, 
# 'bi_bool': True,
# 'uni_layers': 2,
# 'bi_units': 160, 
# 'uni_units': 96,
# 'final_layer_units': 41,
# 'dense_units':39    


def get_gru_parameters():
    GRU = {
        'model_name': None,
        'preprocessing_args': {
            'raw': False,
            'stemming': StemmingOptions.ISRILIGHT,
            'stop_words': StopWordsOptions.REMOVE
        },
        'training_args': {
            'EPOCHS': 38,
            'BATCH_SIZE': 64,
            'emb_dim': 320,  
            'dropout': 0.6164959,
            'learning_rate': 0.003007, 
            'bi_bool': True,
            'uni_layers': 1,
            'bi_units': 72, 
            'uni_units': 104,
            'final_layer_units': 11,
            'dense_units':60     
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
        'model_name': "aubmindlab/bert-base-arabertv01",
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
