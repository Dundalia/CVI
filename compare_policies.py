import argparse
import numpy as np
import sys


def load_policy(filepath: str) -> np.ndarray:
    """Load a policy from a .npy file."""
    try:
        policy = np.load(filepath)
        return policy
    except Exception as e:
        print(f"Error loading policy from {filepath}: {e}")
        sys.exit(1)


def compare_policies(policy1: np.ndarray, policy2: np.ndarray) -> float:
    """Compare two policies and return the percentage of differing actions."""
    if policy1.shape != policy2.shape:
        print(f"Error: Policies have different shapes: {policy1.shape} vs {policy2.shape}")
        sys.exit(1)
    
    n_states = len(policy1)
    differing_states = np.sum(policy1 != policy2)
    percentage_differing = (differing_states / n_states) * 100
    
    return percentage_differing


def main():
    parser = argparse.ArgumentParser(
        description='Compare two policies saved as .npy files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" Examples: python compare_policies.py pi_policy.npy cvi_policy.npy"""
    )
    
    parser.add_argument('policy1', help='Path to first policy .npy file')
    parser.add_argument('policy2', help='Path to second policy .npy file')
    
    args = parser.parse_args()
    
    policy1 = load_policy(args.policy1)
    policy2 = load_policy(args.policy2)
    
    percentage_differing = compare_policies(policy1, policy2)
    
    print(f"Policies comparison:")
    print(f"  Policy 1: {args.policy1} (shape: {policy1.shape})")
    print(f"  Policy 2: {args.policy2} (shape: {policy2.shape})")
    print(f"  Percentage of differing actions: {percentage_differing:.2f}%")


if __name__ == '__main__':
    main()