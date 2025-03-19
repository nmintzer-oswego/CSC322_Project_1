#!/usr/bin/env python3
"""
model_identifier.py - LLM Model Identification Based on Mathematical Error Patterns

This script identifies which Large Language Model is behind an API by analyzing its
unique mathematical error patterns, using a minimum number of API calls. It implements
a hierarchical testing approach that starts with a small screening test set and
dynamically adds more tests if needed for higher confidence.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our modules
from test_generator import TestCaseGenerator
from api_client import APIClient
from fingerprint_module import ModelFingerprint, FingerprintComparison, FingerprintDatabase


class ModelIdentifier:
    """
    Main class for identifying LLMs based on mathematical error patterns.
    """
    
    def __init__(self, 
                api_url: str = "http://localhost:11434/api/generate",
                database_path: str = "model_fingerprints.json",
                cache_dir: str = ".cache",
                confidence_threshold: float = 0.85):
        """
        Initialize the model identifier.
        
        Args:
            api_url: URL of the API endpoint
            database_path: Path to the fingerprint database
            cache_dir: Directory for caching API responses
            confidence_threshold: Threshold for confident identification
        """
        self.api_client = APIClient(api_url=api_url, cache_dir=cache_dir)
        self.test_generator = TestCaseGenerator()
        self.fp_database = FingerprintDatabase(database_path=database_path)
        self.confidence_threshold = confidence_threshold
        
        # Stats for analysis
        self.total_api_calls = 0
        self.identification_history = []
    
    def train_fingerprint(self, model_name: str, verbose: bool = True) -> ModelFingerprint:
        """
        Generate a fingerprint for a known model.
        
        Args:
            model_name: Name of the model
            verbose: Whether to print progress information
            
        Returns:
            Generated fingerprint
        """
        if verbose:
            print(f"\nGenerating fingerprint for model: {model_name}")
            print("=" * 50)
        
        # Generate comprehensive test suite
        tests = self.test_generator.generate_comprehensive_tests(num_tests_per_category=5)
        
        if verbose:
            print(f"Running {len(tests)} comprehensive tests...")
        
        # Run tests and collect results
        results = self.api_client.run_tests(model_name, tests)
        
        # Track API calls
        self.total_api_calls += len([r for r in results if not r.get("cached", False)])
        
        # Generate fingerprint from results
        fingerprint = ModelFingerprint(model_name)
        fingerprint.generate_from_results(results)
        
        # Save to database
        self.fp_database.add_fingerprint(fingerprint)
        self.fp_database.save()
        
        if verbose:
            print("Fingerprint generation complete")
            print(f"Accuracy rate: {fingerprint.fingerprint['accuracy_rate']*100:.2f}%")
            print(f"Error growth rate: {fingerprint.fingerprint.get('error_growth_rate', 'N/A')}")
            print(f"Direction bias: {fingerprint.fingerprint.get('direction_bias', 'N/A')}")
            print("Test type performance:")
            for test_type, stats in fingerprint.fingerprint.get('test_type_performance', {}).items():
                print(f"  {test_type}: {stats['accuracy']*100:.2f}% accuracy, {stats['avg_error']:.2f}% avg error")
        
        return fingerprint
    
    def identify_model(self, unknown_model: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Identify an unknown model by comparing its fingerprint to known models.
        
        Args:
            unknown_model: Name of the unknown model
            verbose: Whether to print progress information
            
        Returns:
            Identification results dictionary
        """
        if verbose:
            print(f"\nIdentifying model: {unknown_model}")
            print("=" * 50)
        
        # Check if we have any fingerprints to compare against
        if not self.fp_database.fingerprints:
            raise ValueError("No fingerprints in database. Please train some models first.")
        
        # Stage 1: Initial screening with a small test set
        if verbose:
            print("Stage 1: Initial screening")
        
        screening_tests = self.test_generator.generate_screening_tests(num_tests=10)
        screening_results = self.api_client.run_tests(unknown_model, screening_tests)
        
        # Track API calls
        self.total_api_calls += len([r for r in screening_results if not r.get("cached", False)])
        
        # Generate preliminary fingerprint
        unknown_fp = ModelFingerprint(unknown_model)
        unknown_fp.generate_from_results(screening_results)
        
        # Compare with known fingerprints
        comparison = FingerprintComparison(self.fp_database.fingerprints)
        initial_matches = comparison.compare(unknown_fp)
        
        # Track identification history
        self.identification_history.append({
            "stage": "screening",
            "tests_run": len(screening_tests),
            "matches": initial_matches[:3]  # Top 3 matches
        })
        
        # Check if we have a confident match already
        top_match, top_score, _ = initial_matches[0]
        
        if verbose:
            print(f"Initial top match: {top_match} (score: {top_score:.4f})")
            for model, score, _ in initial_matches[1:3]:
                print(f"  {model}: {score:.4f}")
        
        if top_score >= self.confidence_threshold:
            if verbose:
                print(f"Confident match found with initial screening: {top_match}")
            
            return {
                "identified_model": top_match,
                "confidence": top_score,
                "tests_run": len(screening_tests),
                "api_calls": self.total_api_calls,
                "all_matches": initial_matches
            }
        
        # Stage 2: Targeted testing for similar models
        if verbose:
            print("\nStage 2: Targeted testing")
        
        # Find top competing models (similar scores)
        competing_models = []
        for model, score, _ in initial_matches:
            if score > top_score - 0.2:  # Within 0.2 of top score
                competing_models.append(model)
        
        if verbose:
            print(f"Running targeted tests for competing models: {', '.join(competing_models)}")
        
        # Get fingerprints for competing models
        competing_fps = {}
        for model in competing_models:
            fp = self.fp_database.get_fingerprint(model)
            if fp:
                competing_fps[model] = fp
        
        if len(competing_fps) > 1:
            # Generate targeted tests for competing models
            targeted_tests = []
            for i in range(len(competing_fps) - 1):
                model1 = competing_models[i]
                for j in range(i+1, len(competing_fps)):
                    model2 = competing_models[j]
                    fp1 = competing_fps[model1].fingerprint
                    fp2 = competing_fps[model2].fingerprint
                    
                    # Generate tests that differentiate these two models
                    model_tests = self.test_generator.generate_targeted_tests(fp1, fp2, num_tests=5)
                    targeted_tests.extend(model_tests)
            
            # Run targeted tests
            targeted_results = self.api_client.run_tests(unknown_model, targeted_tests)
            
            # Track API calls
            self.total_api_calls += len([r for r in targeted_results if not r.get("cached", False)])
            
            # Add results to unknown fingerprint
            all_results = screening_results + targeted_results
            unknown_fp.generate_from_results(all_results)
            
            # Compare again with known fingerprints
            final_matches = comparison.compare(unknown_fp)
            
            # Track identification history
            self.identification_history.append({
                "stage": "targeted",
                "tests_run": len(targeted_tests),
                "matches": final_matches[:3]  # Top 3 matches
            })
            
            top_match, top_score, _ = final_matches[0]
            
            if verbose:
                print(f"Final top match: {top_match} (score: {top_score:.4f})")
                for model, score, _ in final_matches[1:3]:
                    print(f"  {model}: {score:.4f}")
            
            return {
                "identified_model": top_match,
                "confidence": top_score,
                "tests_run": len(screening_tests) + len(targeted_tests),
                "api_calls": self.total_api_calls,
                "all_matches": final_matches
            }
        else:
            # Not enough competing models for targeted testing
            return {
                "identified_model": top_match,
                "confidence": top_score,
                "tests_run": len(screening_tests),
                "api_calls": self.total_api_calls,
                "all_matches": initial_matches
            }
    
    def generate_report(self, identification_result: Dict[str, Any], filename: str = None) -> str:
        """
        Generate a comprehensive identification report.
        
        Args:
            identification_result: Results from identify_model
            filename: Optional filename to save report to
            
        Returns:
            Report text
        """
        identified_model = identification_result["identified_model"]
        confidence = identification_result["confidence"]
        tests_run = identification_result["tests_run"]
        api_calls = identification_result["api_calls"]
        all_matches = identification_result["all_matches"]
        
        report = f"# LLM Model Identification Report\n\n"
        report += f"## Identification Results\n\n"
        report += f"**Identified Model:** {identified_model}\n\n"
        report += f"**Confidence:** {confidence:.2%}\n\n"
        report += f"**Tests Run:** {tests_run}\n\n"
        report += f"**API Calls:** {api_calls}\n\n"
        
        # Add similarity scores for top matches
        report += f"## Top Matches\n\n"
        report += f"| Rank | Model | Similarity Score | Components |\n"
        report += f"|------|-------|------------------|------------|\n"
        
        for i, (model, score, components) in enumerate(all_matches[:5]):
            comp_str = ", ".join([f"{k} ({v:.2f})" for k, v in components.items()])
            report += f"| {i+1} | {model} | {score:.4f} | {comp_str} |\n"
        
        # Add identification history
        if self.identification_history:
            report += f"\n## Identification Process\n\n"
            
            for i, stage in enumerate(self.identification_history):
                stage_name = stage["stage"].capitalize()
                tests_run = stage["tests_run"]
                matches = stage["matches"]
                
                report += f"### Stage {i+1}: {stage_name}\n\n"
                report += f"Tests run: {tests_run}\n\n"
                report += f"Top matches:\n\n"
                
                for model, score, _ in matches:
                    report += f"- {model}: {score:.4f}\n"
                
                report += "\n"
        
        # Add fingerprint analysis if we can get the fingerprints
        identified_fp = self.fp_database.get_fingerprint(identified_model)
        
        if identified_fp:
            report += f"\n## Fingerprint Analysis for {identified_model}\n\n"
            
            # Error by digit
            report += f"### Error by Digit Length\n\n"
            report += f"| Digit Length | Relative Error (%) |\n"
            report += f"|--------------|-------------------|\n"
            
            for digit, error in sorted(identified_fp.fingerprint.get('error_by_digit', {}).items()):
                report += f"| {digit} | {error:.2f} |\n"
            
            # Direction bias
            direction_bias = identified_fp.fingerprint.get('direction_bias')
            if direction_bias is not None:
                report += f"\n### Direction Bias: {direction_bias:.3f}\n"
                if direction_bias > 0:
                    report += "(Model tends to overestimate results)\n"
                else:
                    report += "(Model tends to underestimate results)\n"
            
            # Test type performance
            test_perf = identified_fp.fingerprint.get('test_type_performance', {})
            if test_perf:
                report += f"\n### Performance on Test Types\n\n"
                report += f"| Test Type | Accuracy | Avg Relative Error |\n"
                report += f"|-----------|----------|--------------------|\n"
                
                for test_type, stats in test_perf.items():
                    test_name = test_type.replace('_', ' ').title()
                    report += f"| {test_name} | {stats['accuracy']*100:.2f}% | {stats['avg_error']:.2f}% |\n"
        
        # Save to file if requested
        if filename:
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Report saved to {filename}")
        
        return report


def check_available_models():
    """Check which models are available in the local Ollama instance."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            available_models = [model['name'] for model in response.json().get('models', [])]
            return available_models
        else:
            print(f"Error: Ollama API returned status code {response.status_code}")
            return []
    except Exception as e:
        print(f"Error connecting to Ollama API: {e}")
        return []


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="LLM Model Identification based on Mathematical Error Patterns")
    
    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train fingerprints for known models")
    train_parser.add_argument("models", nargs="+", help="Names of models to train")
    
    # Identify command
    identify_parser = subparsers.add_parser("identify", help="Identify an unknown model")
    identify_parser.add_argument("model", help="Name of the model to identify")
    identify_parser.add_argument("--report", help="Save identification report to file")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List known models in the fingerprint database")
    
    # Common arguments
    parser.add_argument("--api-url", default="http://localhost:11434/api/generate", help="URL of the API endpoint")
    parser.add_argument("--database", default="model_fingerprints.json", help="Path to the fingerprint database")
    parser.add_argument("--cache-dir", default=".cache", help="Directory for caching API responses")
    parser.add_argument("--confidence", type=float, default=0.85, help="Confidence threshold for identification")
    
    args = parser.parse_args()
    
    # Initialize the model identifier
    identifier = ModelIdentifier(
        api_url=args.api_url,
        database_path=args.database,
        cache_dir=args.cache_dir,
        confidence_threshold=args.confidence
    )
    
    if args.command == "train":
        # Check if models are available
        available_models = check_available_models()
        
        for model in args.models:
            if available_models and model not in available_models:
                print(f"Warning: Model '{model}' not found in available Ollama models.")
                print(f"Available models: {', '.join(available_models)}")
                
                proceed = input(f"Do you want to proceed with training {model} anyway? (y/n): ")
                if proceed.lower() != 'y':
                    continue
            
            try:
                identifier.train_fingerprint(model)
                print(f"Successfully trained fingerprint for {model}")
            except Exception as e:
                print(f"Error training fingerprint for {model}: {e}")
    
    elif args.command == "identify":
        # Check if model is available
        available_models = check_available_models()
        
        if available_models and args.model not in available_models:
            print(f"Warning: Model '{args.model}' not found in available Ollama models.")
            print(f"Available models: {', '.join(available_models)}")
            
            proceed = input(f"Do you want to proceed with identifying {args.model} anyway? (y/n): ")
            if proceed.lower() != 'y':
                return
        
        try:
            result = identifier.identify_model(args.model)
            
            print("\nIdentification Results:")
            print(f"Identified as: {result['identified_model']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Tests run: {result['tests_run']}")
            print(f"API calls: {result['api_calls']}")
            
            # Generate report if requested
            if args.report:
                identifier.generate_report(result, args.report)
        
        except Exception as e:
            print(f"Error identifying model: {e}")
    
    elif args.command == "list":
        models = identifier.fp_database.list_models()
        
        if not models:
            print("No models in the fingerprint database. Use 'train' command to add models.")
        else:
            print("\nModels in fingerprint database:")
            for model in sorted(models):
                fp = identifier.fp_database.get_fingerprint(model)
                if fp and 'accuracy_rate' in fp.fingerprint:
                    print(f"- {model} (accuracy rate: {fp.fingerprint['accuracy_rate']*100:.2f}%)")
                else:
                    print(f"- {model}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    import requests  # For checking available models
    main()
