from typing import Any

from .iterables import batched


def dataset_split(
    test_size: float, random_state: int | None = None, **kwargs: Any
) -> dict:
    """
    Split named arrays into aligned train and test dictionaries.
    """
    try:
        import sklearn.model_selection as sk
    except ImportError as error:
        raise ImportError(
            "dataset_split requires the optional 'scikit-learn' dependency."
        ) from error

    if not kwargs:
        raise ValueError("At least one named dataset must be provided.")

    values = list(kwargs.values())
    split_values = sk.train_test_split(
        *values,
        test_size=test_size,
        random_state=random_state,
    )

    output: dict[str, dict[str, Any]] = {"train": {}, "test": {}}

    for (name, _), (train_data, test_data) in zip(
        kwargs.items(), batched(split_values, 2)
    ):
        output["train"][name] = train_data
        output["test"][name] = test_data

    return output
