from tensorflow.keras.utils import to_categorical  # type: ignore
from itertools import islice
import sklearn.model_selection as sk


def batched(iterable, n: int):  # Function is present in itertools for python 3.12+
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def dataset_split(test_size: float, random_state: float, **kwargs) -> dict:
    values = list(kwargs.values())

    splitted = sk.train_test_split(*values, test_size=test_size, random_state=random_state)

    output = {
        'train': dict(), 'test': dict()
    }

    for (k, v), (train_data, test_data) in zip(kwargs.items(), batched(splitted, 2)):
        output['train'][k] = train_data
        output['test'][k] = test_data

    return output