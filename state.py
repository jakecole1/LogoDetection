from pydantic import BaseModel

class State(BaseModel):
    """
    State of the graph
    """
    artwork_file: str
    artwork_slices_folder: str = None
    reference_logo_file: str
    reference_logo_slices_folder: str = None
    model: str
    system_prompt: str
    logos_embeddings: list = None
    artwork_slices_embeddings: list = None
    matches: str = None
    top_k_results: list = None
    top_n_results: list = None
    n: int = 15
    k: int = 10
    max_matches_per_logo: int = 2  # Maximum matches per logo
