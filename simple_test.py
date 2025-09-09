import os
import json
import glob
import sys
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append('.')

from logo_agent import build_graph
from state import State

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "photon-services-f0d3ec1417d0.json"

def load_ground_truth(labels_file: str = "Test_artworks/labels.json") -> Dict[str, List[str]]:
    """Load ground truth labels from JSON file."""
    try:
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        print(f"Loaded ground truth labels for {len(labels)} test artworks")
        return labels
    except FileNotFoundError:
        print(f"Error: Labels file {labels_file} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing labels file: {e}")
        return {}


def get_test_artwork_files(test_dir: str = "Test_artworks") -> List[str]:
    """Get list of PNG files in test artworks directory."""
    png_files = glob.glob(os.path.join(test_dir, "*.png"))
    return sorted(png_files)


def extract_artwork_name(filepath: str) -> str:
    """Extract artwork name from filepath (e.g., 'a1' from 'a1.pdf.png')."""
    filename = os.path.basename(filepath)
    if filename.endswith('.pdf.png'):
        return filename[:-8]  # Remove '.pdf.png' (8 characters)
    elif filename.endswith('.png'):
        return filename[:-4]  # Remove '.png'
    return filename


def parse_vlm_response(vlm_response) -> List[str]:
    """Parse VLM response to extract detected logo names."""
    detected_logos = []
    
    try:
        # Handle different response types
        if isinstance(vlm_response, dict):
            # If it's a dictionary, look for the actual response
            if 'matches' in vlm_response:
                actual_response = vlm_response['matches']
            else:
                actual_response = vlm_response
        else:
            actual_response = vlm_response
        
        # Get the text content
        if hasattr(actual_response, 'text'):
            # If it's a method, call it
            if callable(actual_response.text):
                response_text = actual_response.text()
            else:
                response_text = actual_response.text
        elif hasattr(actual_response, 'content'):
            response_text = actual_response.content
        else:
            response_text = str(actual_response)
        
        print(f"VLM Response: {response_text}")
        
        # Try to parse JSON response
        if '{' in response_text and '}' in response_text:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
            
            try:
                response_data = json.loads(json_str)
                if isinstance(response_data, dict):
                    for key, value in response_data.items():
                        if key != 'are_matches' and value is True:
                            detected_logos.append(key)
            except json.JSONDecodeError:
                pass
        
        # Fallback: look for common logo names in the text
        common_logos = ['E METROLOGIC', 'QR CODE', 'FSC', 'TRIMAN', 'COSIGNE DE TRI']
        for logo in common_logos:
            if logo.lower() in response_text.lower():
                detected_logos.append(logo)
                
    except Exception as e:
        print(f"Error parsing VLM response: {e}")
    
    return detected_logos


def calculate_metrics(ground_truth: List[str], detected: List[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1-score."""
    ground_truth_set = set(ground_truth)
    detected_set = set(detected)
    
    tp = len(ground_truth_set.intersection(detected_set))
    fp = len(detected_set - ground_truth_set)
    fn = len(ground_truth_set - detected_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def test_single_artwork(artwork_file: str, ground_truth_logos: List[str]) -> Dict[str, Any]:
    """Test logo detection on a single artwork file."""
    artwork_name = extract_artwork_name(artwork_file)
    print(f"\n{'='*50}")
    print(f"Testing artwork: {artwork_name}")
    print(f"File: {artwork_file}")
    print(f"Ground truth logos: {ground_truth_logos}")
    
    # Create state for this test
    state_instance = State(
        artwork_file=artwork_file,
        artwork_slices_folder=f"test_slices_{artwork_name}",
        reference_logo_file="reference_pdf_loreal.pdf",
        model="gemini-2.5-pro",
        system_prompt="You are a helpful assistant that can match reference logos to images.",
        k=5,
        n=15
    )
    
    try:
        # Build and run the graph
        app = build_graph(state_instance)
        result = app.invoke(state_instance)
        
        # Extract detected logos from VLM response
        detected_logos = parse_vlm_response(result.get('matches', {}))
        
        print(f"Detected logos: {detected_logos}")
        
        # Calculate metrics for this artwork
        metrics = calculate_metrics(ground_truth_logos, detected_logos)
        
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1']:.3f}")
        
        return {
            'artwork_name': artwork_name,
            'artwork_file': artwork_file,
            'ground_truth_logos': ground_truth_logos,
            'detected_logos': detected_logos,
            'metrics': metrics,
            'success': True
        }
        
    except Exception as e:
        print(f"Error testing {artwork_name}: {e}")
        return {
            'artwork_name': artwork_name,
            'artwork_file': artwork_file,
            'ground_truth_logos': ground_truth_logos,
            'detected_logos': [],
            'metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0},
            'success': False,
            'error': str(e)
        }


def main():
    """Main function to run the logo detection tests."""
    print("Simple Logo Detection Test")
    print("=" * 50)
    
    # Load ground truth labels
    ground_truth = load_ground_truth()
    if not ground_truth:
        print("No ground truth labels loaded. Exiting.")
        return
    
    # Get test artwork files
    test_files = get_test_artwork_files()
    print(f"Found {len(test_files)} test artwork files")
    
    if not test_files:
        print("No test artwork files found!")
        return
    
    # Run tests
    all_results = []
    total_tp = total_fp = total_fn = 0
    
    for artwork_file in test_files:
        artwork_name = extract_artwork_name(artwork_file)
        ground_truth_logos = ground_truth.get(artwork_name, [])
        
        result = test_single_artwork(artwork_file, ground_truth_logos)
        all_results.append(result)
        
        # Accumulate metrics
        total_tp += result['metrics']['tp']
        total_fp += result['metrics']['fp']
        total_fn += result['metrics']['fn']
    
    # Calculate overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total test artworks: {len(all_results)}")
    print(f"Overall Precision: {overall_precision:.3f}")
    print(f"Overall Recall: {overall_recall:.3f}")
    print(f"Overall F1-Score: {overall_f1:.3f}")
    
    # Print individual results
    print("\nIndividual Results:")
    print("-" * 50)
    for result in all_results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['artwork_name']}: "
              f"GT={result['ground_truth_logos']}, "
              f"Detected={result['detected_logos']}, "
              f"F1={result['metrics']['f1']:.3f}")
    
    # Save results to JSON
    results_data = {
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        },
        'individual_results': all_results
    }
    
    with open('simple_test_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: simple_test_results.json")
    print("Test completed!")


if __name__ == "__main__":
    main()
