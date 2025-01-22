from tensorflow.keras.utils import to_categorical
from itertools import islice
import sklearn.model_selection as sk


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def dataset_split(test_size: float, random_state: float, max_number_of_peaks: int, **kwargs):
    values = list(kwargs.values())

    splitted = sk.train_test_split(*values, test_size=test_size, random_state=random_state)

    output = {
        'train': dict(), 'test': dict()
    }

    for (k, v), (train_data, test_data) in zip(kwargs.items(), batched(splitted, 2)):
        if k == 'num_peaks':
            train_data = to_categorical(train_data, max_number_of_peaks + 1)
            test_data = to_categorical(test_data, max_number_of_peaks + 1)

        output['train'][k] = train_data
        output['test'][k] = test_data

    return output