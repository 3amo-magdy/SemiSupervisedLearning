from enum import Enum, auto

class StemmingOptions(Enum):
    NONE = auto()
    ISRILIGHT = auto()
    ISRI = auto()
    ARABICLIGHT = auto()

class StopWordsOptions(Enum):
    KEEP = auto()
    REMOVE = auto()


MARBERT = {
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
    }
}
