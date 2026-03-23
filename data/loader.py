import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Loads dataset exactly as used in the notebook.

    - No additional transformations
    - Keeps raw structure intact
    - Delegates preprocessing to preprocessing module
    """
    df = pd.read_csv(path)
    return df
