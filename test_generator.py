#!/usr/bin/env python3
"""
test_generator.py - Test Case Generator for LLM Fingerprinting

This module generates test cases for mathematical operations designed to reveal
distinctive error patterns in large language models. It includes both general test
cases for initial screening and specialized tests designed to differentiate between
similar models.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class TestCaseGenerator:
    """
    Generator for mathematical test cases used in LLM identification.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the test case generator.

        Args:
            seed: Optional random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_screening_tests(self, num_tests: int = 10) -> List[Dict[str, Any]]:
        """
        Generate a small set of highly discriminative tests for initial model screening.

        Args:
            num_tests: Number of tests to generate (default: 10)

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        # Ensure we include a mix of different test types
        num_power_tests = max(1, num_tests // 5)
        num_repeated_tests = max(1, num_tests // 5)
        num_near_power_tests = max(1, num_tests // 5)
        num_standard_tests = max(1, num_tests - (num_power_tests + num_repeated_tests + num_near_power_tests))

        # 1. Powers of 10 (e.g., 1000 × 100)
        test_cases.extend(self._generate_power_of_10_tests(num_power_tests))

        # 2. Repeated digits (e.g., 777 × 777)
        test_cases.extend(self._generate_repeated_digit_tests(num_repeated_tests))

        # 3. Near powers of 10 (e.g., 999 × 99)
        test_cases.extend(self._generate_near_power_of_10_tests(num_near_power_tests))

        # 4. Standard random tests with varying digit lengths
        test_cases.extend(self._generate_standard_tests(num_standard_tests))

        # Shuffle and return
        self.rng.shuffle(test_cases)
        return test_cases

    def generate_comprehensive_tests(self, num_tests_per_category: int = 5) -> List[Dict[str, Any]]:
        """
        Generate a comprehensive test suite covering various mathematical patterns.

        Args:
            num_tests_per_category: Number of tests to generate per category

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        # 1. Standard tests with varying digit lengths
        for digits in range(2, 9):
            test_cases.extend(self._generate_standard_tests(
                num_tests_per_category,
                min_digits=digits,
                max_digits=digits
            ))

        # 2. Powers of 10
        test_cases.extend(self._generate_power_of_10_tests(num_tests_per_category))

        # 3. Repeated digits
        test_cases.extend(self._generate_repeated_digit_tests(num_tests_per_category))

        # 4. Near powers of 10
        test_cases.extend(self._generate_near_power_of_10_tests(num_tests_per_category))

        # 5. Repeating patterns
        test_cases.extend(self._generate_repeating_pattern_tests(num_tests_per_category))

        # 6. Edge cases
        test_cases.extend(self._generate_edge_cases(num_tests_per_category))

        # Shuffle tests to avoid any bias from order
        self.rng.shuffle(test_cases)
        return test_cases

    def generate_targeted_tests(self,
                                model1_fingerprint: Dict[str, Any],
                                model2_fingerprint: Dict[str, Any],
                                num_tests: int = 5) -> List[Dict[str, Any]]:
        """
        Generate tests specifically designed to differentiate between two similar models.

        Args:
            model1_fingerprint: Fingerprint of first model
            model2_fingerprint: Fingerprint of second model
            num_tests: Number of tests to generate

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        # Analyze where models differ the most
        test_types_to_focus = self._identify_differentiating_test_types(model1_fingerprint, model2_fingerprint)
        digit_lengths_to_focus = self._identify_differentiating_digit_lengths(model1_fingerprint, model2_fingerprint)

        tests_per_type = max(1, num_tests // len(test_types_to_focus))

        # Generate tests focused on areas of maximum difference
        for test_type in test_types_to_focus:
            if test_type == "powers_of_10":
                for _ in range(tests_per_type):
                    d1, d2 = self._select_from_focus_digits(digit_lengths_to_focus)
                    num1 = 10 ** d1
                    num2 = 10 ** d2
                    test_cases.append(self._create_test_case(num1, num2, "powers_of_10"))

            elif test_type == "repeated_digits":
                for _ in range(tests_per_type):
                    d1, d2 = self._select_from_focus_digits(digit_lengths_to_focus)
                    digit = self.rng.randint(1, 9)
                    num1 = int(str(digit) * d1)
                    num2 = int(str(digit) * d2)
                    test_cases.append(self._create_test_case(num1, num2, "repeated_digits"))

            elif test_type == "near_powers_of_10":
                for _ in range(tests_per_type):
                    d1, d2 = self._select_from_focus_digits(digit_lengths_to_focus)
                    num1 = 10 ** d1 - 1
                    num2 = 10 ** d2 - 1
                    test_cases.append(self._create_test_case(num1, num2, "near_powers_of_10"))

            elif test_type == "random":
                for _ in range(tests_per_type):
                    d1, d2 = self._select_from_focus_digits(digit_lengths_to_focus)
                    num1 = self.rng.randint(10 ** (d1 - 1), 10 ** d1 - 1)
                    num2 = self.rng.randint(10 ** (d2 - 1), 10 ** d2 - 1)
                    test_cases.append(self._create_test_case(num1, num2, "random"))

            elif test_type == "repeating_patterns":
                for _ in range(tests_per_type):
                    d1, d2 = self._select_from_focus_digits(digit_lengths_to_focus, min_digits=2, max_digits=4)
                    pattern1 = self.rng.randint(10 ** (d1 - 1), 10 ** d1 - 1)
                    pattern2 = self.rng.randint(10 ** (d2 - 1), 10 ** d2 - 1)
                    repeats = self.rng.randint(2, 3)
                    num1 = int(str(pattern1) * repeats)
                    num2 = int(str(pattern2) * repeats)
                    test_cases.append(self._create_test_case(num1, num2, "repeating_patterns"))

        # If we don't have enough tests yet, add some standard ones
        if len(test_cases) < num_tests:
            additional_tests = self._generate_standard_tests(num_tests - len(test_cases))
            test_cases.extend(additional_tests)

        return test_cases

    def _identify_differentiating_test_types(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> List[str]:
        """
        Identify test types that best differentiate between two models.

        Args:
            fp1: Fingerprint of first model
            fp2: Fingerprint of second model

        Returns:
            List of test types to focus on
        """
        test_types = ["random", "powers_of_10", "repeated_digits", "near_powers_of_10", "repeating_patterns"]
        differentiating_types = []

        # Compare test type performance if available
        tp1 = fp1.get('test_type_performance', {})
        tp2 = fp2.get('test_type_performance', {})

        for test_type in test_types:
            if test_type in tp1 and test_type in tp2:
                # Check if there's a significant difference in accuracy
                acc1 = tp1[test_type].get('accuracy', 0)
                acc2 = tp2[test_type].get('accuracy', 0)

                if abs(acc1 - acc2) > 0.1:  # Significant difference threshold
                    differentiating_types.append(test_type)
                    continue

                # Check if there's a significant difference in average error
                err1 = tp1[test_type].get('avg_error', 0)
                err2 = tp2[test_type].get('avg_error', 0)

                # Calculate relative difference
                avg_err = (err1 + err2) / 2
                if avg_err > 0 and abs(err1 - err2) / avg_err > 0.2:  # 20% difference
                    differentiating_types.append(test_type)

        # If no clear differentiators, include a mix of test types
        if not differentiating_types:
            differentiating_types = ["random", "powers_of_10", "repeated_digits"]

        return differentiating_types

    def _identify_differentiating_digit_lengths(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> List[int]:
        """
        Identify digit lengths that best differentiate between two models.

        Args:
            fp1: Fingerprint of first model
            fp2: Fingerprint of second model

        Returns:
            List of digit lengths to focus on
        """
        diff_digits = []

        # Compare error by digit length
        err1 = fp1.get('error_by_digit', {})
        err2 = fp2.get('error_by_digit', {})

        common_digits = set(str(d) for d in err1.keys()).intersection(set(str(d) for d in err2.keys()))

        for digit in common_digits:
            digit_err1 = err1[digit]
            digit_err2 = err2[digit]

            # Calculate relative difference
            avg_err = (digit_err1 + digit_err2) / 2
            if avg_err > 0 and abs(digit_err1 - digit_err2) / avg_err > 0.2:  # 20% difference
                diff_digits.append(int(digit))

        # If no clear differentiators or too few, include a mix of digit lengths
        if len(diff_digits) < 2:
            diff_digits = [3, 4, 5, 6]

        return diff_digits

    def _select_from_focus_digits(self, focus_digits: List[int], min_digits: int = 2, max_digits: int = 8) -> Tuple[
        int, int]:
        """
        Select two digit lengths from the focus digits.

        Args:
            focus_digits: List of digit lengths to focus on
            min_digits: Minimum digit length
            max_digits: Maximum digit length

        Returns:
            Tuple of (d1, d2) digit lengths
        """
        valid_digits = [d for d in focus_digits if min_digits <= d <= max_digits]

        if not valid_digits:
            valid_digits = list(range(min_digits, max_digits + 1))

        if len(valid_digits) == 1:
            return valid_digits[0], valid_digits[0]

        d1 = self.rng.choice(valid_digits)
        d2 = self.rng.choice(valid_digits)
        return d1, d2

    def _generate_standard_tests(self, num_tests: int, min_digits: int = 2, max_digits: int = 8) -> List[
        Dict[str, Any]]:
        """
        Generate standard random multiplication tests.

        Args:
            num_tests: Number of tests to generate
            min_digits: Minimum digit length
            max_digits: Maximum digit length

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        for _ in range(num_tests):
            # Select digit length for this test
            digits = self.rng.randint(min_digits, max_digits)

            # Generate random numbers with the selected digit length
            num1 = self.rng.randint(10 ** (digits - 1), 10 ** digits - 1)
            num2 = self.rng.randint(10 ** (digits - 1), 10 ** digits - 1)

            test_cases.append(self._create_test_case(num1, num2, "random"))

        return test_cases

    def _generate_power_of_10_tests(self, num_tests: int) -> List[Dict[str, Any]]:
        """
        Generate tests with powers of 10.

        Args:
            num_tests: Number of tests to generate

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        # Generate some combination of powers
        for _ in range(num_tests):
            d1 = self.rng.randint(1, 5)
            d2 = self.rng.randint(1, 5)

            num1 = 10 ** d1
            num2 = 10 ** d2

            test_cases.append(self._create_test_case(num1, num2, "powers_of_10"))

        return test_cases

    def _generate_repeated_digit_tests(self, num_tests: int) -> List[Dict[str, Any]]:
        """
        Generate tests with repeated digits.

        Args:
            num_tests: Number of tests to generate

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        for _ in range(num_tests):
            digit = self.rng.randint(1, 9)
            digits = self.rng.randint(2, 5)

            num1 = int(str(digit) * digits)
            num2 = int(str(digit) * digits)

            test_cases.append(self._create_test_case(num1, num2, "repeated_digits"))

        return test_cases

    def _generate_near_power_of_10_tests(self, num_tests: int) -> List[Dict[str, Any]]:
        """
        Generate tests with numbers near powers of 10 (e.g., 99, 999).

        Args:
            num_tests: Number of tests to generate

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        for _ in range(num_tests):
            d1 = self.rng.randint(2, 5)
            d2 = self.rng.randint(2, 5)

            num1 = 10 ** d1 - 1
            num2 = 10 ** d2 - 1

            test_cases.append(self._create_test_case(num1, num2, "near_powers_of_10"))

        return test_cases

    def _generate_repeating_pattern_tests(self, num_tests: int) -> List[Dict[str, Any]]:
        """
        Generate tests with repeating patterns (e.g., 123123).

        Args:
            num_tests: Number of tests to generate

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        for _ in range(num_tests):
            # Create base patterns
            base_digits = self.rng.randint(2, 3)
            pattern1 = self.rng.randint(10 ** (base_digits - 1), 10 ** base_digits - 1)
            pattern2 = self.rng.randint(10 ** (base_digits - 1), 10 ** base_digits - 1)

            # Repeat the patterns
            repeats = self.rng.randint(2, 3)
            num1 = int(str(pattern1) * repeats)
            num2 = int(str(pattern2) * repeats)

            test_cases.append(self._create_test_case(num1, num2, "repeating_patterns"))

        return test_cases

    def _generate_edge_cases(self, num_tests: int) -> List[Dict[str, Any]]:
        """
        Generate edge cases that often reveal model limitations.

        Args:
            num_tests: Number of tests to generate

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        edge_case_generators = [
            # Numbers with lots of carrying operations
            lambda: (999, 999),
            lambda: (9999, 9999),
            # Numbers with many zeros
            lambda: (1001, 1001),
            lambda: (10001, 10001),
            # Numbers with alternating digits
            lambda: (int(''.join(['1' if i % 2 == 0 else '9' for i in range(4)])),
                     int(''.join(['1' if i % 2 == 0 else '9' for i in range(4)]))),
            # Very large single-digit multiplication
            lambda: (9, 9),
            lambda: (9, 99),
            lambda: (99, 999),
            # Palindromic numbers
            lambda: (121, 121),
            lambda: (12321, 12321),
        ]

        # Select num_tests edge cases, with replacement if needed
        for _ in range(num_tests):
            generator = self.rng.choice(edge_case_generators)
            num1, num2 = generator()
            test_cases.append(self._create_test_case(num1, num2, "edge_case"))

        return test_cases

    def _create_test_case(self, num1: int, num2: int, test_type: str) -> Dict[str, Any]:
        """
        Create a test case dictionary from given parameters.

        Args:
            num1: First number
            num2: Second number
            test_type: Type of test

        Returns:
            Test case dictionary
        """
        true_result = num1 * num2
        max_digits = max(len(str(num1)), len(str(num2)))

        return {
            "num1": num1,
            "num2": num2,
            "true_result": true_result,
            "digits": max_digits,
            "test_type": test_type
        }


# Example usage
if __name__ == "__main__":
    # This is just for demonstration
    generator = TestCaseGenerator(seed=42)

    # Generate initial screening tests
    screening_tests = generator.generate_screening_tests(10)
    print(f"Generated {len(screening_tests)} screening tests")

    # Print a few examples
    for i, test in enumerate(screening_tests[:3]):
        print(f"Test {i + 1}: {test['num1']} × {test['num2']} = {test['true_result']} ({test['test_type']})")

    # Generate comprehensive tests
    comprehensive_tests = generator.generate_comprehensive_tests(3)
    print(f"\nGenerated {len(comprehensive_tests)} comprehensive tests")

    # Count by test type
    test_types = {}
    for test in comprehensive_tests:
        test_type = test['test_type']
        test_types[test_type] = test_types.get(test_type, 0) + 1

    print("Test type distribution:")
    for test_type, count in test_types.items():
        print(f"  {test_type}: {count} tests")