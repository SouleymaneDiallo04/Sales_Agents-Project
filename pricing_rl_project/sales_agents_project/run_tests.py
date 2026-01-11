#!/usr/bin/env python3
"""
Script pour exécuter tous les tests
"""

import subprocess
import sys
import os
from pathlib import Path

def run_unit_tests():
    """Exécuter les tests unitaires"""
    print("🧪 Exécution des tests unitaires...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short"
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Erreurs:", result.stderr)

    return result.returncode == 0

def run_integration_tests():
    """Exécuter les tests d'intégration"""
    print("🔗 Exécution des tests d'intégration...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short"
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Erreurs:", result.stderr)

    return result.returncode == 0

def run_all_tests():
    """Exécuter tous les tests"""
    print("🧪 Exécution de tous les tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=sales_agents_project",
        "--cov-report=html"
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Erreurs:", result.stderr)

    return result.returncode == 0

def main():
    """Fonction principale"""
    print("🧪 Suite de tests Sales Agents Project")
    print("=" * 40)

    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "unit":
            success = run_unit_tests()
        elif test_type == "integration":
            success = run_integration_tests()
        else:
            print("Usage: python run_tests.py [unit|integration|all]")
            return
    else:
        success = run_all_tests()

    if success:
        print("✅ Tous les tests sont passés!")
        sys.exit(0)
    else:
        print("❌ Certains tests ont échoué")
        sys.exit(1)

if __name__ == "__main__":
    main()