#!/usr/bin/env python3
"""
Test script to verify MiniZinc installation and basic functionality
"""

import subprocess
import sys
import os

def test_minizinc_installation():
    """Test if MiniZinc is properly installed"""
    try:
        # Test if minizinc command is available
        result = subprocess.run(['minizinc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ MiniZinc is installed and working")
            print(f"Version: {result.stdout.strip()}")
            return True
        else:
            print("✗ MiniZinc command failed")
            print(f"Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("✗ MiniZinc command not found. Please install MiniZinc.")
        return False
    except subprocess.TimeoutExpired:
        print("✗ MiniZinc command timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing MiniZinc: {e}")
        return False

def test_solver_availability():
    """Test if Gecode solver is available"""
    try:
        # Test if gecode solver is available
        result = subprocess.run(['minizinc', '--solvers'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            if 'gecode' in result.stdout.lower():
                print("✓ Gecode solver is available")
                return True
            else:
                print("✗ Gecode solver not found")
                print("Available solvers:")
                print(result.stdout)
                return False
        else:
            print("✗ Failed to list solvers")
            return False
            
    except Exception as e:
        print(f"✗ Error testing solver availability: {e}")
        return False

def test_simple_model():
    """Test a simple MiniZinc model"""
    try:
        # Create a simple test model
        test_model = """
% Simple test model
var 1..3: x;
var 1..3: y;
constraint x + y = 5;
solve satisfy;
output [show(x), " ", show(y)];
"""
        
        # Write model to temporary file
        with open('test_model.mzn', 'w') as f:
            f.write(test_model)
        
        # Run the model
        result = subprocess.run(['minizinc', '--solver', 'gecode', 'test_model.mzn'], 
                              capture_output=True, text=True, timeout=30)
        
        # Clean up
        os.remove('test_model.mzn')
        
        if result.returncode == 0:
            print("✓ Simple model solved successfully")
            print(f"Solution: {result.stdout.strip()}")
            return True
        else:
            print("✗ Simple model failed")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing simple model: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing MiniZinc installation...")
    print("=" * 50)
    
    tests = [
        ("MiniZinc Installation", test_minizinc_installation),
        ("Solver Availability", test_solver_availability),
        ("Simple Model", test_simple_model)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! MiniZinc is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check MiniZinc installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 