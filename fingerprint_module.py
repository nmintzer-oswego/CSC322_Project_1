#!/usr/bin/env python3
"""
fingerprint.py - LLM Mathematical Error Fingerprinting Module

This module handles the creation, analysis, and comparison of mathematical
error fingerprints for Large Language Models. It defines the core functionality
for generating fingerprints from test results and comparing fingerprints to
identify models based on their unique error patterns.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import os
from collections import defaultdict


class ModelFingerprint:
    """
    Class representing a mathematical error fingerprint for a language model.
    Contains methods for analyzing test results and creating fingerprint
    features that uniquely identify a model.
    """

    def __init__(self, model_name: str = None, fingerprint_data: Dict = None):
        """
        Initialize a ModelFingerprint instance.
        
        Args:
            model_name: Name of the model
            fingerprint_data: Optional pre-existing fingerprint data to load
        """
        self.model_name = model_name
        
        # Initialize fingerprint components
        if fingerprint_data:
            self.fingerprint = fingerprint_data
        else:
            self.fingerprint = {
                'model_name': model_name,
                'error_by_digit': {},                # Error rates by digit length
                'direction_bias': None,              # Tendency to over/underestimate
                'error_growth_rate': None,           # How quickly error grows with complexity
                'test_type_performance': {},         # Performance on different test types
                'digit_error_patterns': {},          # Errors by digit position
                'accuracy_rate': None,               # Overall accuracy rate
                'distinguishing_features': []        # Key features that distinguish this model
            }
        
        # Raw test results used to create the fingerprint
        self.results = []
    
    def generate_from_results(self, results: List[Dict[str, Any]]) -> Dict:
        """
        Generate a fingerprint from a list of test results.
        
        Args:
            results: List of test result dictionaries
            
        Returns:
            Complete fingerprint dictionary
        """
        self.results = results
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Skip rows with missing values for error calculation
        valid_data = df.dropna(subset=['relative_error'])
        
        if valid_data.empty:
            print(f"Warning: No valid results found for model {self.model_name}")
            return self.fingerprint
        
        # 1. Error rates by digit length
        error_by_digit = valid_data.groupby('digits')['relative_error'].mean().to_dict()
        self.fingerprint['error_by_digit'] = error_by_digit
        
        # 2. Error growth rates between digit lengths
        growth_rates = {}
        for i in range(3, 9):  # Starting from 3-digit calculations
            if i-1 in error_by_digit and i in error_by_digit:
                prev_error = error_by_digit[i-1]
                curr_error = error_by_digit[i]
                
                if prev_error > 0:
                    growth_rates[i] = curr_error / prev_error
        
        self.fingerprint['growth_rates'] = growth_rates
        self.fingerprint['error_growth_rate'] = np.mean(list(growth_rates.values())) if growth_rates else None
        
        # 3. Direction bias (tendency to over/underestimate)
        true_vs_predicted = valid_data[['true_result', 'predicted_result']].dropna()
        if not true_vs_predicted.empty:
            true_vs_predicted['error_direction'] = true_vs_predicted.apply(
                lambda row: 1 if row['predicted_result'] > row['true_result'] else 
                         (-1 if row['predicted_result'] < row['true_result'] else 0), 
                axis=1
            )
            self.fingerprint['direction_bias'] = true_vs_predicted['error_direction'].mean()
        
        # 4. Performance on different test types
        type_performance = {}
        for test_type in valid_data['test_type'].unique():
            type_data = valid_data[valid_data['test_type'] == test_type]
            if not type_data.empty:
                type_performance[test_type] = {
                    'accuracy': (type_data['absolute_error'] == 0).mean(),
                    'avg_error': type_data['relative_error'].mean()
                }
        
        self.fingerprint['test_type_performance'] = type_performance
        
        # 5. Overall accuracy rate
        self.fingerprint['accuracy_rate'] = (valid_data['absolute_error'] == 0).mean()
        
        # 6. Digit-level error patterns
        self._analyze_digit_error_patterns(valid_data)
        
        return self.fingerprint
    
    def _analyze_digit_error_patterns(self, valid_data: pd.DataFrame) -> None:
        """
        Analyze digit-level error patterns.
        
        Args:
            valid_data: DataFrame with valid test results
        """
        # Filter to only include results with digit errors data
        digit_results = valid_data.dropna(subset=['digit_errors'])
        
        if digit_results.empty:
            return
        
        # Count of errors by digit position (from right)
        position_errors = defaultdict(lambda: {'count': 0, 'total': 0, 'error_sum': 0})
        
        for _, row in digit_results.iterrows():
            errors = row['digit_errors']
            if isinstance(errors, list):
                for i, error in enumerate(errors):
                    position = len(errors) - i - 1  # Position from right
                    position_errors[position]['total'] += 1
                    if error != 0:
                        position_errors[position]['count'] += 1
                        position_errors[position]['error_sum'] += error
        
        # Calculate error rate and average error by position
        digit_error_patterns = {}
        for pos, stats in position_errors.items():
            if stats['total'] > 0:
                error_rate = stats['count'] / stats['total']
                avg_error = stats['error_sum'] / stats['count'] if stats['count'] > 0 else 0
                digit_error_patterns[pos] = {
                    'error_rate': error_rate,
                    'avg_error': avg_error
                }
        
        self.fingerprint['digit_error_patterns'] = digit_error_patterns
    
    def identify_distinguishing_features(self, other_fingerprints: List['ModelFingerprint']) -> List[str]:
        """
        Identify features that distinguish this fingerprint from others.
        
        Args:
            other_fingerprints: List of other ModelFingerprint objects to compare against
            
        Returns:
            List of distinguishing feature descriptions
        """
        distinguishing_features = []
        
        # Skip if no fingerprint data
        if not self.fingerprint or not other_fingerprints:
            return distinguishing_features
        
        # 1. Check error growth rate
        if self.fingerprint.get('error_growth_rate'):
            other_rates = [fp.fingerprint.get('error_growth_rate') for fp in other_fingerprints 
                         if fp.fingerprint.get('error_growth_rate') is not None]
            
            if other_rates:
                avg_other_rate = np.mean(other_rates)
                if abs(self.fingerprint['error_growth_rate'] - avg_other_rate) > 0.5:
                    distinguishing_features.append(
                        f"Distinctive error growth rate: {self.fingerprint['error_growth_rate']:.2f}x "
                        f"(vs. average {avg_other_rate:.2f}x)"
                    )
        
        # 2. Check direction bias
        if self.fingerprint.get('direction_bias') is not None:
            other_biases = [fp.fingerprint.get('direction_bias') for fp in other_fingerprints 
                          if fp.fingerprint.get('direction_bias') is not None]
            
            if other_biases and all(np.sign(self.fingerprint['direction_bias']) != np.sign(bias) for bias in other_biases):
                bias_direction = "overestimate" if self.fingerprint['direction_bias'] > 0 else "underestimate"
                distinguishing_features.append(
                    f"Unique error direction bias: Tends to {bias_direction} "
                    f"(bias: {self.fingerprint['direction_bias']:.2f})"
                )
        
        # 3. Check digit-specific error jumps
        growth_rates = self.fingerprint.get('growth_rates', {})
        for digit, growth in growth_rates.items():
            other_digit_rates = []
            for fp in other_fingerprints:
                other_growth = fp.fingerprint.get('growth_rates', {}).get(digit)
                if other_growth is not None:
                    other_digit_rates.append(other_growth)
            
            if other_digit_rates and all(abs(growth - other_rate) > 1.0 for other_rate in other_digit_rates):
                distinguishing_features.append(
                    f"Distinctive error jump at {digit} digits: {growth:.2f}x"
                )
        
        # 4. Check test type performance
        test_perf = self.fingerprint.get('test_type_performance', {})
        for test_type, stats in test_perf.items():
            other_accuracies = []
            for fp in other_fingerprints:
                other_test_perf = fp.fingerprint.get('test_type_performance', {}).get(test_type, {})
                if 'accuracy' in other_test_perf:
                    other_accuracies.append(other_test_perf['accuracy'])
            
            if other_accuracies and all(abs(stats['accuracy'] - acc) > 0.2 for acc in other_accuracies):
                performance = "good" if stats['accuracy'] > np.mean(other_accuracies) else "poor"
                distinguishing_features.append(
                    f"Distinctive {performance} performance on {test_type.replace('_', ' ')} tests: "
                    f"{stats['accuracy']*100:.1f}% accuracy"
                )
        
        self.fingerprint['distinguishing_features'] = distinguishing_features
        return distinguishing_features

    def to_dict(self) -> Dict:
        """Convert fingerprint to dictionary"""
        return self.fingerprint
    
    def to_json(self) -> str:
        """Convert fingerprint to JSON string"""
        return json.dumps(self.fingerprint, indent=2, default=self._json_serialize)
    
    def _json_serialize(self, obj):
        """Helper method to serialize numpy types for JSON"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)


class FingerprintComparison:
    """
    Class for comparing fingerprints to identify models.
    """
    
    def __init__(self, known_fingerprints: Dict[str, ModelFingerprint] = None):
        """
        Initialize with known fingerprints.
        
        Args:
            known_fingerprints: Dictionary mapping model names to fingerprint objects
        """
        self.known_fingerprints = known_fingerprints or {}
    
    def compare(self, unknown_fingerprint: ModelFingerprint) -> List[Tuple[str, float, Dict]]:
        """
        Compare an unknown fingerprint against known fingerprints.
        
        Args:
            unknown_fingerprint: Fingerprint to identify
            
        Returns:
            List of (model_name, similarity_score, component_scores) tuples,
            sorted by similarity score (highest first)
        """
        similarity_scores = []
        
        for model_name, known_fp in self.known_fingerprints.items():
            score, components = self._calculate_similarity(unknown_fingerprint, known_fp)
            similarity_scores.append((model_name, score, components))
        
        # Sort by similarity score, highest first
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return similarity_scores
    
    def _calculate_similarity(self, fp1: ModelFingerprint, fp2: ModelFingerprint) -> Tuple[float, Dict[str, float]]:
        """
        Calculate similarity score between two fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            (overall_similarity_score, component_scores)
        """
        components = {}
        
        # 1. Compare error growth rates
        if (fp1.fingerprint.get('error_growth_rate') is not None and 
            fp2.fingerprint.get('error_growth_rate') is not None):
            
            fp1_growth = fp1.fingerprint['error_growth_rate']
            fp2_growth = fp2.fingerprint['error_growth_rate']
            
            if fp2_growth > 0:
                growth_similarity = 1 - min(abs(fp1_growth - fp2_growth) / fp2_growth, 1)
                components['growth_rate'] = growth_similarity
        
        # 2. Compare direction bias
        if (fp1.fingerprint.get('direction_bias') is not None and 
            fp2.fingerprint.get('direction_bias') is not None):
            
            fp1_bias = fp1.fingerprint['direction_bias']
            fp2_bias = fp2.fingerprint['direction_bias']
            
            # Check if sign matches
            sign_match = np.sign(fp1_bias) == np.sign(fp2_bias)
            
            # Normalize the magnitude difference
            bias_similarity = 1 - min(abs(fp1_bias - fp2_bias) / (abs(fp2_bias) + 0.01), 1)
            
            # Adjust similarity based on sign match
            bias_similarity = bias_similarity * 0.5 + (0.5 if sign_match else 0)
            components['direction_bias'] = bias_similarity
        
        # 3. Compare error by digit patterns
        if fp1.fingerprint.get('error_by_digit') and fp2.fingerprint.get('error_by_digit'):
            fp1_errors = fp1.fingerprint['error_by_digit']
            fp2_errors = fp2.fingerprint['error_by_digit']
            
            # Find common digit lengths
            common_digits = set(fp1_errors.keys()).intersection(set(fp2_errors.keys()))
            
            if common_digits:
                digit_similarities = []
                for digit in common_digits:
                    fp1_error = fp1_errors[digit]
                    fp2_error = fp2_errors[digit]
                    
                    if fp2_error > 0:
                        # Similarity based on relative difference
                        similarity = 1 - min(abs(fp1_error - fp2_error) / fp2_error, 1)
                        digit_similarities.append(similarity)
                
                if digit_similarities:
                    components['error_pattern'] = np.mean(digit_similarities)
        
        # 4. Compare special test type performance
        if fp1.fingerprint.get('test_type_performance') and fp2.fingerprint.get('test_type_performance'):
            fp1_perf = fp1.fingerprint['test_type_performance']
            fp2_perf = fp2.fingerprint['test_type_performance']
            
            # Find common test types
            common_types = set(fp1_perf.keys()).intersection(set(fp2_perf.keys()))
            
            if common_types:
                type_similarities = []
                for test_type in common_types:
                    # Compare accuracy
                    fp1_acc = fp1_perf[test_type]['accuracy']
                    fp2_acc = fp2_perf[test_type]['accuracy']
                    
                    acc_similarity = 1 - abs(fp1_acc - fp2_acc)
                    
                    # Compare average error
                    fp1_err = fp1_perf[test_type]['avg_error']
                    fp2_err = fp2_perf[test_type]['avg_error']
                    
                    if fp2_err > 0:
                        err_similarity = 1 - min(abs(fp1_err - fp2_err) / fp2_err, 1)
                    else:
                        err_similarity = 1 - min(abs(fp1_err - fp2_err), 1)
                    
                    # Combined similarity for this test type
                    type_similarity = (acc_similarity + err_similarity) / 2
                    type_similarities.append(type_similarity)
                
                if type_similarities:
                    components['special_types'] = np.mean(type_similarities)
        
        # 5. Compare digit-level error patterns
        if fp1.fingerprint.get('digit_error_patterns') and fp2.fingerprint.get('digit_error_patterns'):
            fp1_patterns = fp1.fingerprint['digit_error_patterns']
            fp2_patterns = fp2.fingerprint['digit_error_patterns']
            
            common_positions = set(fp1_patterns.keys()).intersection(set(fp2_patterns.keys()))
            
            if common_positions:
                position_similarities = []
                for pos in common_positions:
                    # Compare error rates
                    fp1_rate = fp1_patterns[pos]['error_rate']
                    fp2_rate = fp2_patterns[pos]['error_rate']
                    
                    rate_similarity = 1 - abs(fp1_rate - fp2_rate)
                    
                    # Compare average errors
                    fp1_avg = fp1_patterns[pos]['avg_error']
                    fp2_avg = fp2_patterns[pos]['avg_error']
                    
                    if abs(fp2_avg) > 0:
                        avg_similarity = 1 - min(abs(fp1_avg - fp2_avg) / abs(fp2_avg), 1)
                    else:
                        avg_similarity = 1 - min(abs(fp1_avg - fp2_avg), 1)
                    
                    position_similarity = (rate_similarity + avg_similarity) / 2
                    position_similarities.append(position_similarity)
                
                if position_similarities:
                    components['digit_positions'] = np.mean(position_similarities)
        
        # Calculate overall similarity score (weighted average of components)
        weights = {
            'growth_rate': 0.3,
            'direction_bias': 0.2,
            'error_pattern': 0.3,
            'special_types': 0.1,
            'digit_positions': 0.1
        }
        
        # Only consider components that exist
        available_weights = {comp: weight for comp, weight in weights.items() if comp in components}
        
        total_weight = sum(available_weights.values())
        
        if total_weight > 0:
            # Normalize weights
            norm_weights = {comp: weight/total_weight for comp, weight in available_weights.items()}
            
            # Calculate weighted score
            score = sum(components[comp] * norm_weights[comp] for comp in components.keys())
        else:
            score = 0
        
        return score, components


class FingerprintDatabase:
    """
    Class for managing a database of model fingerprints.
    """
    
    def __init__(self, database_path: str = "model_fingerprints.json"):
        """
        Initialize the fingerprint database.
        
        Args:
            database_path: Path to the fingerprint database file
        """
        self.database_path = database_path
        self.fingerprints = {}
        
        # Load existing database if it exists
        self.load()
    
    def load(self) -> Dict[str, ModelFingerprint]:
        """
        Load fingerprints from the database file.
        
        Returns:
            Dictionary of model fingerprints
        """
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    data = json.load(f)
                
                for model_name, fp_data in data.items():
                    self.fingerprints[model_name] = ModelFingerprint(model_name, fp_data)
                
                print(f"Loaded {len(self.fingerprints)} fingerprints from {self.database_path}")
            except Exception as e:
                print(f"Error loading fingerprint database: {e}")
        
        return self.fingerprints
    
    def save(self) -> bool:
        """
        Save fingerprints to the database file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert fingerprints to dictionaries
            data = {name: fp.to_dict() for name, fp in self.fingerprints.items()}
            
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_serialize)
            
            print(f"Saved {len(self.fingerprints)} fingerprints to {self.database_path}")
            return True
        except Exception as e:
            print(f"Error saving fingerprint database: {e}")
            return False
    
    def add_fingerprint(self, fingerprint: ModelFingerprint) -> None:
        """
        Add or update a fingerprint in the database.
        
        Args:
            fingerprint: ModelFingerprint object to add
        """
        self.fingerprints[fingerprint.model_name] = fingerprint
    
    def get_fingerprint(self, model_name: str) -> Optional[ModelFingerprint]:
        """
        Get a fingerprint from the database.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelFingerprint object or None if not found
        """
        return self.fingerprints.get(model_name)
    
    def remove_fingerprint(self, model_name: str) -> bool:
        """
        Remove a fingerprint from the database.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if successful, False otherwise
        """
        if model_name in self.fingerprints:
            del self.fingerprints[model_name]
            return True
        return False
    
    def list_models(self) -> List[str]:
        """
        Get a list of all model names in the database.
        
        Returns:
            List of model names
        """
        return list(self.fingerprints.keys())
    
    def _json_serialize(self, obj):
        """Helper method to serialize numpy types for JSON"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)


# Example usage
if __name__ == "__main__":
    # This is just for demonstration
    # Create a sample fingerprint
    fp = ModelFingerprint("sample_model")
    
    # Generate from test results (this would come from actual tests)
    sample_results = [
        {
            "model": "sample_model",
            "num1": 123,
            "num2": 456,
            "true_result": 56088,
            "predicted_result": 56088,
            "absolute_error": 0,
            "relative_error": 0.0,
            "digits": 3,
            "test_type": "random",
            "digit_errors": [0, 0, 0, 0, 0],
            "digit_error_count": 0
        },
        # ... more test results would go here
    ]
    
    fp.generate_from_results(sample_results)
    
    # Save to database
    db = FingerprintDatabase()
    db.add_fingerprint(fp)
    db.save()
    
    # Compare fingerprints
    unknown_fp = ModelFingerprint("unknown")
    unknown_fp.generate_from_results(sample_results)  # For demonstration
    
    comparison = FingerprintComparison(db.fingerprints)
    results = comparison.compare(unknown_fp)
    
    for model, score, components in results:
        print(f"Model: {model}, Score: {score:.4f}")
        for comp, comp_score in components.items():
            print(f"  {comp}: {comp_score:.4f}")
