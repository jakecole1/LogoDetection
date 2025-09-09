from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import List
from utils.make_slices import MakeSlices
import fitz
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.extract_logos import ExtractLogos
from utils.match_cv import match_cv
from state import State
from vertexai.vision_models import MultiModalEmbeddingModel, Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob
from vertexai.generative_models import Part


def load_and_slice(State: State) -> State:
    """
    Load the images and slice them
    """
    print(f"Loading and slicing artwork: {State.artwork_file}")
    slicer = MakeSlices(client_name="")
    metadata = slicer.create_nxn_blocks(
        file_path=State.artwork_file,
        output_folder_path=State.artwork_slices_folder,
        block_size=500,
        zoom_factor=2.0
    )
    print(f"Created {len(metadata)} slices in {State.artwork_slices_folder}")
    return {"artwork_slices_folder": State.artwork_slices_folder}

def embedding_comparison(State: State) -> State:
    """
    Compare Vertex multimodal image embeddings between logos and artwork slices and return top-k results.
    """
    print("Generating embeddings for logos and artwork slices...")
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

    
    logos_folder = "extracted_logos"
    if not os.path.exists(logos_folder):
        raise ValueError(f"No logos folder found at {logos_folder}")

    logo_files = sorted(glob.glob(os.path.join(logos_folder, "*.png")) +
                        glob.glob(os.path.join(logos_folder, "*.jpg")))

    if not logo_files:
        raise ValueError(f"No logo images found in {logos_folder}")

    logo_vecs, logo_meta = [], []
    print(f"Processing {len(logo_files)} logo files...")
    for fp in logo_files:
        try:
            img = Image.load_from_file(fp)
            emb = model.get_embeddings(image=img, contextual_text=None, dimension=1408)
            vec = np.array(emb.image_embedding, dtype=np.float32)  # (1408,)
            logo_vecs.append(vec)                                   
            logo_meta.append(fp)
        except Exception as e:
            print(f"Error processing logo {fp}: {e}")

    if not logo_vecs:
        raise ValueError("Failed to produce any logo embeddings")

    logos_mat = np.stack(logo_vecs, axis=0)  # (L, 1408)

    
    artwork_files = sorted(glob.glob(os.path.join(State.artwork_slices_folder, "*.png")) +
                           glob.glob(os.path.join(State.artwork_slices_folder, "*.jpg")))

    if not artwork_files:
        raise ValueError(f"No artwork slice images found in {State.artwork_slices_folder}")

    art_vecs, art_meta = [], []
    print(f"Processing {len(artwork_files)} artwork slice files...")
    for fp in artwork_files:
        try:
            img = Image.load_from_file(fp)
            emb = model.get_embeddings(image=img, contextual_text=None, dimension=1408)
            vec = np.array(emb.image_embedding, dtype=np.float32)
            art_vecs.append(vec)                                   
            art_meta.append(fp)
        except Exception as e:
            print(f"Error processing artwork slice {fp}: {e}")

    if not art_vecs:
        raise ValueError("Failed to produce any artwork embeddings")

    artwork_mat = np.stack(art_vecs, axis=0)  # (A, 1408)
    def l2norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n
    logos_mat = l2norm(logos_mat)
    artwork_mat = l2norm(artwork_mat)

    print("Computing similarity between logos and artwork slices...")
    S = cosine_similarity(logos_mat, artwork_mat)  # (num_logos, num_slices)

    # Get top N logos overall with 1-2 matches each
    n = getattr(State, "n", 5)  # Default to top 5 logos
    max_matches_per_logo = getattr(State, "max_matches_per_logo", 2)  # Default to 2 matches per logo
    
    # Find the best match for each logo
    logo_best_scores = []
    for li in range(S.shape[0]):
        sims = S[li]
        best_score = np.max(sims)
        best_artwork_idx = np.argmax(sims)
        logo_best_scores.append({
            "logo_index": li,
            "best_score": best_score,
            "best_artwork_idx": best_artwork_idx
        })
    
    # Sort logos by their best match score and take top N
    logo_best_scores.sort(key=lambda x: x["best_score"], reverse=True)
    top_n_logos = logo_best_scores[:n]
    
    top_k_results = []
    for logo_info in top_n_logos:
        li = logo_info["logo_index"]
        sims = S[li]
        
        # Get top matches for this logo (up to max_matches_per_logo)
        top_idx = np.argpartition(-sims, kth=min(max_matches_per_logo-1, len(sims)-1))[:max_matches_per_logo]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        matches = []
        for rank, ai in enumerate(top_idx, start=1):
            matches.append({
                "logo_index": int(li),
                "logo_file": logo_meta[li],
                "artwork_slice_index": int(ai),
                "artwork_file": art_meta[ai],
                "similarity_score": float(sims[ai]),
                "rank": rank,
            })
        top_k_results.append({
            "logo_index": int(li),
            "logo_file": logo_meta[li],
            "top_matches": matches
        })

    print(f"Found top {n} logos overall with up to {max_matches_per_logo} matches each")

    return {
        "logos_embeddings": logos_mat,                 # shape (L, 1408)
        "artwork_slices_embeddings": artwork_mat,      # shape (A, 1408)
        "top_k_results": top_k_results
    }       

def reference_logos(State: State) -> State:
    """
    Extract the reference logos from the images
    """
    reference_logos = fitz.open(State.reference_logo_file)[0]
    return {"reference_logos": reference_logos}


class gemini_vlm(BaseModel):
    model: str
    system_prompt: str
    
    def __init__(self, system_prompt: str, model: str, **kwargs):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            **kwargs
        )
        self._client = ChatGoogleGenerativeAI(model=model, temperature=0)
    
    @property
    def client(self):
        return self._client
    
    def vlm_matching(self, State: State) -> State:
        """
        Match the reference logos to the images
        """ 
        from langchain_core.messages import HumanMessage
        
        
        results_to_use = State.top_n_results if State.top_n_results is not None else State.top_k_results
        if results_to_use is None:
            print("No results are provided. Using reference logo file instead.")
        
        # Prepare the text content
        text_content = """Please use these logo images as a reference. Attached to each logo image is a BRIGHT RED LABEL which is the name of the reference logo. 
If a given logo is in one of the artwork slices, please return the name of the reference logo in the following format:
{"logo_name": True} with logo_name being the name of the reference logo (BRIGHT RED LABEL). if you see any matching logos, return {"are_matches": True}
If you do not see any matching logos, return {"are_matches": False}
Do not return anything else. 
Here are the reference logos to match the artwork slices to."""

        # Prepare content list for multimodal message
        content = [{"type": "text", "text": text_content}]
        
        # Add reference logo images
        if State.reference_logo_slices_folder and os.path.exists(State.reference_logo_slices_folder):
            logo_files = glob.glob(os.path.join(State.reference_logo_slices_folder, "*.png")) + glob.glob(os.path.join(State.reference_logo_slices_folder, "*.jpg"))
            for logo_file in logo_files:
                with open(logo_file, "rb") as image_file:
                    img_bytes = image_file.read()
                content.append({
                    "type": "media",
                    "mime_type": "image/png",
                    "data": img_bytes
                })
        elif State.reference_logo_file:
            with open(State.reference_logo_file, "rb") as image_file:
                img_bytes = image_file.read()
            content.append({
                "type": "media",
                "mime_type": "application/pdf",
                "data": img_bytes
            })
        else:
            raise ValueError("Reference logo slices folder not found or empty")

        content.append({
            "type": "text",
            "text": "Here are the artwork slices to match the reference logos to."
        })
        
        # Add artwork slice images
        if State.artwork_slices_folder and os.path.exists(State.artwork_slices_folder):
            artwork_files = glob.glob(os.path.join(State.artwork_slices_folder, "*.png")) + glob.glob(os.path.join(State.artwork_slices_folder, "*.jpg"))
            for artwork_file in artwork_files:
                with open(artwork_file, "rb") as image_file:
                    img_bytes = image_file.read()
                content.append({
                    "type": "media",
                    "mime_type": "image/png" if artwork_file.lower().endswith(".png") else "image/jpeg",
                    "data": img_bytes
                })
        else:
            raise ValueError("Artwork slices folder not found or empty")

        # Create the message
        message = HumanMessage(content=content)
        
        # Invoke with the message and system prompt
        try:
            matches = self.client.invoke([message])
            print(f"VLM matching completed. Response: {matches.text}")
            return {"matches": matches}
        except Exception as e:
            print(f"Error in VLM matching: {e}")
            return {"matches": None, "error": str(e)}


def extract_logos_wrapper(State: State) -> State:
    """
    Wrapper function for ExtractLogos to work with LangGraph
    """
    extract_logos = ExtractLogos(
        pdf_path=State.reference_logo_file,
        page_number=0,
        dpi=300,
        out_dir="extracted_logos",
        state=State
    )
    result = extract_logos.extract_logos()
    return {"reference_logo_slices_folder": "extracted_logos"}


def build_graph(state_instance: State):
    vlm = gemini_vlm(system_prompt=state_instance.system_prompt, model=state_instance.model)
    graph = StateGraph(State)  
    graph.add_node("load_and_slice", load_and_slice)
    graph.add_node("reference_logos", reference_logos)
    graph.add_node("extract_logos", extract_logos_wrapper)
    graph.add_node("embedding_comparison", embedding_comparison)
    graph.add_node("match_cv", match_cv)
    graph.add_node("vlm_matching", vlm.vlm_matching)

    graph.set_entry_point("load_and_slice")
    graph.add_edge("load_and_slice", "reference_logos")
    graph.add_edge("reference_logos", "extract_logos")
    graph.add_edge("extract_logos", "embedding_comparison")
    
    # Conditionally add match_cv based on whether n is set
    if state_instance.n is not None:
        graph.add_edge("embedding_comparison", "match_cv")
        graph.add_edge("match_cv", "vlm_matching")
    else:
        graph.add_edge("embedding_comparison", "vlm_matching")
    
    graph.add_edge("vlm_matching", END)

    app = graph.compile()
    return app


def llm_only_graph(state_instance: State):
    vlm = gemini_vlm(system_prompt=state_instance.system_prompt, model=state_instance.model)
    graph = StateGraph(State)  
    graph.add_node("load_and_slice", load_and_slice)
    graph.add_node("reference_logos", reference_logos)
    graph.add_node("vlm_matching", vlm.vlm_matching)

    graph.set_entry_point("load_and_slice")
    graph.add_edge("load_and_slice", "reference_logos")
    graph.add_edge("reference_logos", "vlm_matching")
    graph.add_edge("vlm_matching", END)

    app = graph.compile()
    return app

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "photon-services-f0d3ec1417d0.json"
    state_instance = State(artwork_file=r"Test_artworks\a9.pdf", artwork_slices_folder="artwork_slices", reference_logo_file="reference_pdf_loreal.pdf", model="gemini-2.5-pro", system_prompt="You are a helpful assistant that can match reference logos to images.")
    #app = llm_only_graph(state_instance)
    app = build_graph(state_instance)
    result = app.invoke(state_instance)