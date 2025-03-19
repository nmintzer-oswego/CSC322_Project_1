#!/usr/bin/env python3
"""
api_client.py - API Client for LLM Model Identification

This module provides a client for interacting with LLM APIs (specifically Ollama)
with caching to minimize API calls. It handles sending test cases to the API,
parsing responses, and extracting numerical answers.
"""

import requests
import json
import time
import re
import os
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class APIClient:
    """
    Client for interacting with LLM APIs with caching.
    """

    def __init__(self, api_url: str = "http://localhost:11434/api/generate",
                 cache_dir: str = ".cache", max_retries: int = 3,
                 retry_delay: float = 2.0):
        """
        Initialize the API client.

        Args:
            api_url: URL of the API endpoint
            cache_dir: Directory for caching API responses
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_url = api_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Set up cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache hit/miss stats
        self.cache_hits = 0
        self.cache_misses = 0

    def query_model(self, model: str, prompt: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Send a query to the API and get the response.

        Args:
            model: Name of the model to query
            prompt: Prompt to send to the model
            use_cache: Whether to use the cache

        Returns:
            Response dictionary
        """
        if use_cache:
            # Check cache first
            cache_key = self._generate_cache_key(model, prompt)
            cached_response = self._get_from_cache(cache_key)

            if cached_response:
                self.cache_hits += 1
                return cached_response

        self.cache_misses += 1

        # Prepare payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        # Send request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, json=payload)

                if response.status_code == 200:
                    result = response.json()

                    # Cache the response
                    if use_cache:
                        self._save_to_cache(cache_key, result)

                    return result
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    time.sleep(self.retry_delay)
            except Exception as e:
                print(f"Request error: {e}")
                time.sleep(self.retry_delay)

        # Failed after all retries
        return {"error": "Failed to get response after multiple attempts"}

    def run_test(self, model: str, test_case: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
        """
        Run a specific test case on a model.

        Args:
            model: Name of the model to test
            test_case: Test case dictionary
            use_cache: Whether to use the cache

        Returns:
            Test result dictionary
        """
        num1 = test_case["num1"]
        num2 = test_case["num2"]
        test_type = test_case.get("test_type", "random")

        # Construct prompt
        prompt = self._generate_test_prompt(num1, num2)

        # Query the model
        api_response = self.query_model(model, prompt, use_cache)

        # Process response
        if "error" in api_response:
            return {
                "model": model,
                "num1": num1,
                "num2": num2,
                "true_result": test_case["true_result"],
                "predicted_result": None,
                "absolute_error": None,
                "relative_error": None,
                "digits": test_case["digits"],
                "test_type": test_type,
                "digit_errors": None,
                "digit_error_count": None,
                "raw_response": api_response.get("error", "Unknown error"),
                "cached": api_response.get("cached", False)
            }

        response_text = api_response.get("response", "")

        # Extract number from response
        predicted_result = self._extract_number(response_text)

        # Calculate error metrics if prediction was obtained
        if predicted_result is not None:
            true_result = test_case["true_result"]
            absolute_error = abs(predicted_result - true_result)
            relative_error = (absolute_error / true_result) * 100 if true_result != 0 else float('inf')

            # Analyze digit-by-digit error pattern
            digit_errors = self._analyze_digit_errors(predicted_result, true_result)

            return {
                "model": model,
                "num1": num1,
                "num2": num2,
                "true_result": true_result,
                "predicted_result": predicted_result,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "digits": test_case["digits"],
                "test_type": test_type,
                "digit_errors": digit_errors,
                "digit_error_count": sum(1 for e in digit_errors if e != 0),
                "raw_response": response_text,
                "cached": api_response.get("cached", False)
            }
        else:
            return {
                "model": model,
                "num1": num1,
                "num2": num2,
                "true_result": test_case["true_result"],
                "predicted_result": None,
                "absolute_error": None,
                "relative_error": None,
                "digits": test_case["digits"],
                "test_type": test_type,
                "digit_errors": None,
                "digit_error_count": None,
                "raw_response": response_text,
                "cached": api_response.get("cached", False)
            }

    def run_tests(self, model: str, test_cases: List[Dict[str, Any]], use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Run multiple test cases on a model.

        Args:
            model: Name of the model to test
            test_cases: List of test case dictionaries
            use_cache: Whether to use the cache

        Returns:
            List of test result dictionaries
        """
        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"Running test {i}/{len(test_cases)} for model {model}", end="\r")
            result = self.run_test(model, test_case, use_cache)
            results.append(result)

        print(f"Completed {len(test_cases)} tests for model {model} "
              f"(cache hits: {self.cache_hits}, misses: {self.cache_misses})")

        return results

    def _generate_test_prompt(self, num1: int, num2: int) -> str:
        """
        Generate a prompt for a multiplication test.

        Args:
            num1: First number
            num2: Second number

        Returns:
            Prompt string
        """
        return f"""Please calculate the exact result of multiplying these two numbers:
{num1} × {num2}
Provide only the numerical result without any explanation."""

    def _extract_number(self, text: str) -> Optional[int]:
        """
        Extract a numerical answer from model response text.

        Args:
            text: Response text

        Returns:
            Extracted number or None if not found
        """
        # Clean text - remove commas and spaces from numbers
        text = text.replace(",", "").replace(" ", "")

        # Try to find patterns that suggest a calculation result
        result_patterns = [
            r'result(?:\s+is)?\s*[:=]?\s*(\d+)',
            r'answer(?:\s+is)?\s*[:=]?\s*(\d+)',
            r'product(?:\s+is)?\s*[:=]?\s*(\d+)',
            r'(?:\s|^)=\s*(\d+)',
            r'equals\s*(\d+)'
        ]

        for pattern in result_patterns:
            matches = re.search(pattern, text.lower())
            if matches:
                return int(matches.group(1))

        # If no specific result pattern found, look for all numbers
        numbers = re.findall(r'\d+', text)

        if numbers:
            # Take the largest number found (assuming it's the result)
            return int(max(numbers, key=len))

        return None

    def _analyze_digit_errors(self, predicted: int, true: int) -> List[int]:
        """
        Analyze errors on a digit-by-digit basis.

        Args:
            predicted: Predicted number
            true: True number

        Returns:
            List of digit errors (0 means correct digit)
        """
        # Convert to strings for digit comparison
        pred_str = str(predicted)
        true_str = str(true)

        # Initialize error array with zeros (0 means correct digit)
        max_len = max(len(pred_str), len(true_str))
        errors = [0] * max_len

        # Pad the shorter number with leading zeros
        pred_str = pred_str.zfill(max_len)
        true_str = true_str.zfill(max_len)

        # Compare each digit and record differences
        for i in range(max_len):
            if pred_str[i] != true_str[i]:
                # Record the difference between digits
                errors[i] = int(pred_str[i]) - int(true_str[i])

        return errors

    def _generate_cache_key(self, model: str, prompt: str) -> str:
        """
        Generate a cache key for a model-prompt pair.

        Args:
            model: Model name
            prompt: Prompt text

        Returns:
            Cache key string
        """
        # Create a unique hash based on the model and prompt
        content = f"{model}:{prompt}"
        hash_obj = hashlib.md5(content.encode())
        return hash_obj.hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response if it exists.

        Args:
            cache_key: Cache key

        Returns:
            Cached response or None if not found
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                # Mark as coming from cache
                cached_data["cached"] = True
                return cached_data
            except Exception as e:
                print(f"Error reading from cache: {e}")

        return None

    def _save_to_cache(self, cache_key: str, response: Dict[str, Any]) -> None:
        """
        Save a response to the cache.

        Args:
            cache_key: Cache key
            response: Response to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
        except Exception as e:
            print(f"Error writing to cache: {e}")

    def get_cache_stats(self) -> Tuple[int, int]:
        """
        Get cache hit/miss statistics.

        Returns:
            Tuple of (hits, misses)
        """
        return self.cache_hits, self.cache_misses


# Example usage
if __name__ == "__main__":
    # This is just for demonstration
    client = APIClient()

    # Test a simple calculation
    model = "llama3.2:1b"  # Adjust based on what's available in your Ollama instance
    test_case = {
        "num1": 123,
        "num2": 456,
        "true_result": 56088,
        "digits": 3,
        "test_type": "random"
    }

    print(f"Testing {model} with {test_case['num1']} × {test_case['num2']}")
    result = client.run_test(model, test_case)

    # Display result
    if result["predicted_result"] is not None:
        print(f"Model predicted: {result['predicted_result']}")
        print(f"True result: {result['true_result']}")
        print(f"Relative error: {result['relative_error']:.2f}%")
        if result["cached"]:
            print("(Result retrieved from cache)")
    else:
        print(f"Model failed to provide a valid numerical answer")
        print(f"Raw response: {result['raw_response'][:100]}...")

    # Try a second call to demonstrate caching
    print("\nRunning same test again to demonstrate caching:")
    result2 = client.run_test(model, test_case)

    if result2["cached"]:
        print("Successfully retrieved from cache!")

    # Cache stats
    hits, misses = client.get_cache_stats()
    print(f"\nCache stats: {hits} hits, {misses} misses")