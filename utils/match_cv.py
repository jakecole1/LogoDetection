import cv2
import os
from typing import Dict, Any
import sys


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from state import State


def match_cv(State: State) -> State:
    """
    LangGraph node that performs ORB matching on top_k_results to filter to top_n_results.
    
    Args:
        State: The current state containing top_k_results
        
    Returns:
        State: Updated state with top_n_results (or unchanged if n not set)
    """
    # Check if n is set - if not, skip ORB matching
    if State.n is None:
        print("ORB matching skipped: n not set in state. Passing through top_k_results as top_n_results.")
        return {"top_n_results": State.top_k_results}
    
    print(f"Starting ORB matching to filter top_k_results to top_n_results (n={State.n})...")
    
    # Check if top_k_results is available
    if State.top_k_results is None:
        raise ValueError("top_k_results not found in state. Make sure embedding_comparison runs before match_cv.")
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Initialize FLANN matcher for ORB descriptors
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                       table_number=6,
                       key_size=12,
                       multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    top_n_results = []
    n = int(State.k)
    
    print(f"Processing {len(State.top_k_results)} logos with ORB matching...")
    
    for logo_result in State.top_k_results:
        logo_file = logo_result['logo_file']
        logo_matches = logo_result['top_matches']
        
        # Load the logo image
        if not os.path.exists(logo_file):
            print(f"Warning: Logo file {logo_file} not found, skipping...")
            continue
            
        logo_img = cv2.imread(logo_file, cv2.IMREAD_GRAYSCALE)
        if logo_img is None:
            print(f"Warning: Could not load logo image {logo_file}, skipping...")
            continue
            
        # Detect ORB features in logo
        logo_kp, logo_desc = orb.detectAndCompute(logo_img, None)
        if logo_desc is None or len(logo_desc) < 10:
            print(f"Warning: Not enough ORB features in logo {logo_file}, skipping...")
            continue
        
        # Score each artwork slice match using ORB
        scored_matches = []
        
        for match in logo_matches:
            artwork_file = match['artwork_file']
            
            # Load the artwork slice image
            if not os.path.exists(artwork_file):
                print(f"Warning: Artwork file {artwork_file} not found, skipping...")
                continue
                
            artwork_img = cv2.imread(artwork_file, cv2.IMREAD_GRAYSCALE)
            if artwork_img is None:
                print(f"Warning: Could not load artwork image {artwork_file}, skipping...")
                continue
            
            # Detect ORB features in artwork slice
            artwork_kp, artwork_desc = orb.detectAndCompute(artwork_img, None)
            if artwork_desc is None or len(artwork_desc) < 10:
                # Not enough features, use original similarity score
                match['orb_score'] = 0.0
                match['orb_matches_count'] = 0
                scored_matches.append(match)
                continue
            
            try:
                # Match descriptors using FLANN
                matches = flann.knnMatch(logo_desc, artwork_desc, k=2)
                
                # Apply Lowe's ratio test to filter good matches
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n_match = match_pair
                        if m.distance < 0.7 * n_match.distance:
                            good_matches.append(m)
                
                # Calculate ORB-based score
                orb_matches_count = len(good_matches)
                orb_score = min(orb_matches_count / 50.0, 1.0)  # Normalize to 0-1, cap at 1.0
                
                # Combine with original similarity score (weighted average)
                original_score = match['similarity_score']
                combined_score = 0.7 * original_score + 0.3 * orb_score
                
                # Update match with ORB information
                match['orb_score'] = orb_score
                match['orb_matches_count'] = orb_matches_count
                match['combined_score'] = combined_score
                
                scored_matches.append(match)
                
            except Exception as e:
                print(f"Warning: Error matching {artwork_file}: {e}")
                # Fallback to original score
                match['orb_score'] = 0.0
                match['orb_matches_count'] = 0
                match['combined_score'] = match['similarity_score']
                scored_matches.append(match)
        
        # Sort by combined score and take top n
        if not scored_matches:
            print(f"Warning: No scored matches for logo {logo_result['logo_index']}, skipping...")
            continue
            
        scored_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Ensure n is a valid integer and within bounds
        n_safe = min(int(n), len(scored_matches))
        top_n_matches = scored_matches[:n_safe]
        
        # Create result for this logo
        logo_result_filtered = {
            "logo_index": logo_result['logo_index'],
            "logo_file": logo_result['logo_file'],
            "top_matches": top_n_matches
        }
        
        top_n_results.append(logo_result_filtered)
        
        print(f"Logo {logo_result['logo_index']}: {len(scored_matches)} candidates -> {len(top_n_matches)} top matches")
    
    print(f"ORB matching completed. Filtered to top {n} results for {len(top_n_results)} logos.")
    
    return {
        "top_n_results": top_n_results
    }


def calculate_orb_similarity(img1_path: str, img2_path: str, orb: cv2.ORB = None) -> Dict[str, Any]:
    """
    Calculate ORB-based similarity between two images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        orb: ORB detector instance (optional)
        
    Returns:
        Dictionary with similarity metrics
    """
    if orb is None:
        orb = cv2.ORB_create(nfeatures=1000)
    
    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return {"orb_score": 0.0, "matches_count": 0, "error": "Could not load images"}
    
    # Detect features
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)
    
    if desc1 is None or desc2 is None or len(desc1) < 10 or len(desc2) < 10:
        return {"orb_score": 0.0, "matches_count": 0, "error": "Not enough features"}
    
    # Match features
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                       table_number=6,
                       key_size=12,
                       multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n_match = match_pair
                if m.distance < 0.7 * n_match.distance:
                    good_matches.append(m)
        
        matches_count = len(good_matches)
        orb_score = min(matches_count / 50.0, 1.0)  # Normalize to 0-1
        
        return {
            "orb_score": orb_score,
            "matches_count": matches_count,
            "total_features_1": len(kp1),
            "total_features_2": len(kp2)
        }
        
    except Exception as e:
        return {"orb_score": 0.0, "matches_count": 0, "error": str(e)}


if __name__ == "__main__":
    # Test the ORB matching functionality
    print("ORB matching module loaded successfully.")
    print("This module provides the match_cv function for use as a LangGraph node.")
