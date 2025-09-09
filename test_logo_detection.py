#!/usr/bin/env python3
"""
Test script for logo detection system using Test_artworks and labels as ground truth.

This script evaluates the logo detection pipeline against known ground truth labels
and provides comprehensive metrics including precision, recall, F1-score, and confusion matrix.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import sys
from pathlib import Path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "photon-services-f0d3ec1417d0.json"
# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(parent_dir)

from logo_agent import build_graph
from state import State


class LogoDetectionTester:
    """
    Test class for evaluating logo detection system against ground truth labels.
    """
    
    def __init__(self, test_artworks_dir: str = "Test_artworks", 
                 labels_file: str = "Test_artworks/labels.json",
                 reference_logo_file: str = "reference_pdf_loreal.pdf",
                 model: str = "gemini-2.5-pro"):
        """
        Initialize the tester with test data paths.
        
        Args:
            test_artworks_dir: Directory containing test artwork PNG files
            labels_file: Path to JSON file with ground truth labels
            reference_logo_file: Path to reference logo PDF file
            model: Model name for the VLM
        """
        self.test_artworks_dir = test_artworks_dir
        self.labels_file = labels_file
        self.reference_logo_file = reference_logo_file
        self.model = model
        self.ground_truth = self._load_ground_truth()
        self.test_results = {}
        self.all_detected_logos = set()
        self.all_ground_truth_logos = set()
        
    def _load_ground_truth(self) -> Dict[str, List[str]]:
        """Load ground truth labels from JSON file."""
        try:
            with open(self.labels_file, 'r') as f:
                labels = json.load(f)
            print(f"Loaded ground truth labels for {len(labels)} test artworks")
            return labels
        except FileNotFoundError:
            print(f"Error: Labels file {self.labels_file} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing labels file: {e}")
            return {}
    
    def _get_test_artwork_files(self) -> List[str]:
        """Get list of PNG files in test artworks directory."""
        png_files = glob.glob(os.path.join(self.test_artworks_dir, "*.png"))
        return sorted(png_files)
    
    def _extract_artwork_name(self, filepath: str) -> str:
        """Extract artwork name from filepath (e.g., 'a1' from 'a1.pdf.png')."""
        filename = os.path.basename(filepath)
        # Remove .pdf.png extension to get base name
        if filename.endswith('.pdf.png'):
            return filename[:-7]  # Remove '.pdf.png'
        elif filename.endswith('.png'):
            return filename[:-4]  # Remove '.png'
        return filename
    
    def run_single_test(self, artwork_file: str) -> Dict[str, Any]:
        """
        Run logo detection on a single artwork file.
        
        Args:
            artwork_file: Path to the artwork PNG file
            
        Returns:
            Dictionary containing test results
        """
        artwork_name = self._extract_artwork_name(artwork_file)
        print(f"\n{'='*50}")
        print(f"Testing artwork: {artwork_name}")
        print(f"File: {artwork_file}")
        
        # Get ground truth labels for this artwork
        ground_truth_logos = self.ground_truth.get(artwork_name, [])
        print(f"Ground truth logos: {ground_truth_logos}")
        
        # Create state for this test
        state_instance = State(
            artwork_file=artwork_file,
            artwork_slices_folder=f"test_slices_{artwork_name}",
            reference_logo_file=self.reference_logo_file,
            model=self.model,
            system_prompt="You are a helpful assistant that can match reference logos to images.",
            k=15,  # Top 15 matches from embedding comparison
            n=5    # Top 5 matches after ORB filtering
        )
        
        try:
            # Build and run the graph
            app = build_graph(state_instance)
            result = app.invoke(state_instance)
            
            # Extract detected logos from VLM response
            detected_logos = self._parse_vlm_response(result.get('matches', {}))
            
            print(f"Detected logos: {detected_logos}")
            
            # Calculate metrics for this artwork
            metrics = self._calculate_metrics(ground_truth_logos, detected_logos)
            
            test_result = {
                'artwork_name': artwork_name,
                'artwork_file': artwork_file,
                'ground_truth_logos': ground_truth_logos,
                'detected_logos': detected_logos,
                'metrics': metrics,
                'raw_vlm_response': str(result.get('matches', {}))
            }
            
            return test_result
            
        except Exception as e:
            print(f"Error testing {artwork_name}: {e}")
            return {
                'artwork_name': artwork_name,
                'artwork_file': artwork_file,
                'ground_truth_logos': ground_truth_logos,
                'detected_logos': [],
                'metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0},
                'error': str(e)
            }
    
    def _parse_vlm_response(self, vlm_response) -> List[str]:
        """
        Parse VLM response to extract detected logo names.
        
        Args:
            vlm_response: VLM response object or string
            
        Returns:
            List of detected logo names
        """
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
            
            # Try to parse JSON response
            if '{' in response_text and '}' in response_text:
                # Extract JSON part
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
                
                try:
                    response_data = json.loads(json_str)
                    if isinstance(response_data, dict):
                        # Extract logo names where value is True
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
    
    def _calculate_metrics(self, ground_truth: List[str], detected: List[str]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score for a single artwork.
        
        Args:
            ground_truth: List of ground truth logo names
            detected: List of detected logo names
            
        Returns:
            Dictionary with metrics
        """
        ground_truth_set = set(ground_truth)
        detected_set = set(detected)
        
        # True positives: logos that are both in ground truth and detected
        tp = len(ground_truth_set.intersection(detected_set))
        
        # False positives: logos detected but not in ground truth
        fp = len(detected_set - ground_truth_set)
        
        # False negatives: logos in ground truth but not detected
        fn = len(ground_truth_set - detected_set)
        
        # Calculate metrics
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
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run logo detection tests on all test artworks.
        
        Returns:
            Dictionary containing all test results and overall metrics
        """
        print("Starting comprehensive logo detection testing...")
        print(f"Test artworks directory: {self.test_artworks_dir}")
        print(f"Ground truth labels: {self.labels_file}")
        
        # Get all test artwork files
        test_files = self._get_test_artwork_files()
        print(f"Found {len(test_files)} test artwork files")
        
        if not test_files:
            print("No test artwork files found!")
            return {}
        
        # Run tests on each artwork
        all_results = []
        for artwork_file in test_files:
            result = self.run_single_test(artwork_file)
            all_results.append(result)
            self.test_results[result['artwork_name']] = result
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(all_results)
        
        # Collect all unique logos
        for result in all_results:
            self.all_detected_logos.update(result['detected_logos'])
            self.all_ground_truth_logos.update(result['ground_truth_logos'])
        
        return {
            'individual_results': all_results,
            'overall_metrics': overall_metrics,
            'all_detected_logos': list(self.all_detected_logos),
            'all_ground_truth_logos': list(self.all_ground_truth_logos)
        }
    
    def _calculate_overall_metrics(self, all_results: List[Dict]) -> Dict[str, float]:
        """
        Calculate overall metrics across all test artworks.
        
        Args:
            all_results: List of individual test results
            
        Returns:
            Dictionary with overall metrics
        """
        total_tp = sum(result['metrics']['tp'] for result in all_results)
        total_fp = sum(result['metrics']['fp'] for result in all_results)
        total_fn = sum(result['metrics']['fn'] for result in all_results)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # Calculate macro averages (average of individual F1 scores)
        macro_f1 = np.mean([result['metrics']['f1'] for result in all_results])
        macro_precision = np.mean([result['metrics']['precision'] for result in all_results])
        macro_recall = np.mean([result['metrics']['recall'] for result in all_results])
        
        return {
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'num_test_artworks': len(all_results)
        }
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "test_report.html") -> str:
        """
        Generate a comprehensive HTML test report.
        
        Args:
            results: Test results dictionary
            output_file: Output HTML file path
            
        Returns:
            Path to generated report file
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Logo Detection Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c5aa0; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Logo Detection Test Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p><strong>Total Test Artworks:</strong> {results['overall_metrics']['num_test_artworks']}</p>
                <p><strong>Total Ground Truth Logos:</strong> {len(results['all_ground_truth_logos'])}</p>
                <p><strong>Total Detected Logos:</strong> {len(results['all_detected_logos'])}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-box">
                    <div class="metric-value">{results['overall_metrics']['overall_precision']:.3f}</div>
                    <div class="metric-label">Overall Precision</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{results['overall_metrics']['overall_recall']:.3f}</div>
                    <div class="metric-label">Overall Recall</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{results['overall_metrics']['overall_f1']:.3f}</div>
                    <div class="metric-label">Overall F1-Score</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{results['overall_metrics']['macro_f1']:.3f}</div>
                    <div class="metric-label">Macro F1-Score</div>
                </div>
            </div>
            
            <h2>Detailed Results by Artwork</h2>
            <table>
                <tr>
                    <th>Artwork</th>
                    <th>Ground Truth</th>
                    <th>Detected</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Status</th>
                </tr>
        """
        
        for result in results['individual_results']:
            status = "Success" if result['metrics']['f1'] > 0.5 else "Needs Improvement"
            status_class = "success" if result['metrics']['f1'] > 0.5 else "error"
            
            html_content += f"""
                <tr>
                    <td>{result['artwork_name']}</td>
                    <td>{', '.join(result['ground_truth_logos']) if result['ground_truth_logos'] else 'None'}</td>
                    <td>{', '.join(result['detected_logos']) if result['detected_logos'] else 'None'}</td>
                    <td>{result['metrics']['precision']:.3f}</td>
                    <td>{result['metrics']['recall']:.3f}</td>
                    <td>{result['metrics']['f1']:.3f}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Logo Coverage Analysis</h2>
            <h3>Ground Truth Logos</h3>
            <ul>
        """
        
        for logo in sorted(results['all_ground_truth_logos']):
            html_content += f"<li>{logo}</li>"
        
        html_content += """
            </ul>
            
            <h3>Detected Logos</h3>
            <ul>
        """
        
        for logo in sorted(results['all_detected_logos']):
            html_content += f"<li>{logo}</li>"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Test report generated: {output_file}")
        return output_file
    
    def save_results_json(self, results: Dict[str, Any], output_file: str = "test_results.json") -> str:
        """
        Save test results to JSON file.
        
        Args:
            results: Test results dictionary
            output_file: Output JSON file path
            
        Returns:
            Path to generated JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Test results saved to: {output_file}")
        return output_file


def main():
    """Main function to run the logo detection tests."""
    print("Logo Detection Test Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = LogoDetectionTester()
    
    # Check if test data exists
    if not os.path.exists(tester.test_artworks_dir):
        print(f"Error: Test artworks directory '{tester.test_artworks_dir}' not found!")
        return
    
    if not os.path.exists(tester.labels_file):
        print(f"Error: Labels file '{tester.labels_file}' not found!")
        return
    
    # Run all tests
    results = tester.run_all_tests()
    
    if not results:
        print("No test results generated!")
        return
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total test artworks: {results['overall_metrics']['num_test_artworks']}")
    print(f"Overall Precision: {results['overall_metrics']['overall_precision']:.3f}")
    print(f"Overall Recall: {results['overall_metrics']['overall_recall']:.3f}")
    print(f"Overall F1-Score: {results['overall_metrics']['overall_f1']:.3f}")
    print(f"Macro F1-Score: {results['overall_metrics']['macro_f1']:.3f}")
    
    # Generate reports
    tester.generate_report(results)
    tester.save_results_json(results)
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
