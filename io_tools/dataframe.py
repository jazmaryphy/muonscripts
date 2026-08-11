# %%
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

import muon_tools.muonic_dataframe as df

# %%
def get_dataframe(
    results: List[Dict[str, Any]], 
    sc_matrix: Optional[list] = None
)-> List[pd.DataFrame]:
    """INFO"""

    return df.get_dataframe(results, sc_matrix)