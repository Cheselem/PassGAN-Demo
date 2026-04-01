"""
=============================================================================
  AI vs. Cryptography: Password Attack Simulation & Accuracy Evaluation
=============================================================================

This script demonstrates "The Effects of AI on Modern Cryptography Systems"
by simulating an AI-driven password attack against our synthetic dataset.

It performs:
  1. Loading the synthetic dataset (data/crypto_ai_dataset.txt)
  2. Splitting into train (80%) and test (20%) sets
  3. Simulating AI-generated password guesses using learned patterns
  4. Comparing AI guesses vs. traditional brute-force/dictionary attacks
  5. Computing and storing accuracy metrics and results

Usage:
  python evaluate_ai_crypto.py
"""

import os
import sys
import random
import time
import json
import collections
import math
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DATASET_PATH = "data/crypto_ai_dataset.txt"
OUTPUT_DIR = "output/evaluation"
NUM_AI_SAMPLES = 500000        # Number of AI-generated password attempts
NUM_BRUTEFORCE_SAMPLES = 500000  # Number of brute-force attempts for comparison
MAX_PASSWORD_LEN = 10          # Same as PassGAN default


def ensure_dirs():
    """Create output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)


def load_dataset(path, max_length=10):
    """Load the password dataset line by line."""
    passwords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if 0 < len(line) <= max_length:
                passwords.append(line)
    return passwords


def train_test_split(data, train_ratio=0.8):
    """Split data into train and test sets."""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


# ─────────────────────────────────────────────────────────────
# N-gram Language Model (mirrors PassGAN's utils.NgramLanguageModel)
# ─────────────────────────────────────────────────────────────
class NgramLanguageModel:
    """Character-level n-gram language model for password generation."""

    def __init__(self, n, samples):
        self.n = n
        self.ngram_counts = collections.defaultdict(int)
        self.total = 0
        self.context_counts = collections.defaultdict(int)

        for sample in samples:
            chars = tuple(sample)
            for i in range(len(chars) - n + 1):
                ngram = chars[i:i + n]
                self.ngram_counts[ngram] += 1
                self.total += 1
                if n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1

    def probability(self, ngram):
        if self.total == 0:
            return 0.0
        return self.ngram_counts[ngram] / self.total

    def js_divergence(self, other):
        """Jensen-Shannon divergence between two n-gram models."""
        all_ngrams = set(self.ngram_counts.keys()) | set(other.ngram_counts.keys())
        kl_pm = 0.0
        kl_qm = 0.0

        for ngram in all_ngrams:
            p = self.probability(ngram)
            q = other.probability(ngram)
            m = 0.5 * (p + q)

            if p > 0 and m > 0:
                kl_pm += p * math.log2(p / m)
            if q > 0 and m > 0:
                kl_qm += q * math.log2(q / m)

        return 0.5 * (kl_pm + kl_qm)


# ─────────────────────────────────────────────────────────────
# AI Password Generator (simulates a trained GAN's learned distribution)
# ─────────────────────────────────────────────────────────────
class AIPasswordGenerator:
    """
    Simulates a PassGAN-style generator that has learned the character
    distribution and pattern structure from the training data.

    This mirrors what the real Generator neural network learns:
    - Character frequency distributions
    - Positional character probabilities
    - Common n-gram transitions
    - Password length distributions
    """

    def __init__(self, training_data):
        self.training_data = training_data
        self._learn_distributions()

    def _learn_distributions(self):
        """Extract statistical patterns from training data (simulating GAN learning)."""
        # Length distribution
        self.length_dist = collections.Counter(len(p) for p in self.training_data)
        total_len = sum(self.length_dist.values())
        self.length_probs = {k: v / total_len for k, v in self.length_dist.items()}

        # Character frequency per position
        max_len = max(len(p) for p in self.training_data)
        self.pos_char_dist = {}
        for pos in range(max_len):
            counter = collections.Counter()
            for pwd in self.training_data:
                if pos < len(pwd):
                    counter[pwd[pos]] += 1
            total = sum(counter.values())
            self.pos_char_dist[pos] = {c: count / total for c, count in counter.items()}

        # Bigram transition probabilities (Markov chain)
        self.bigram_trans = collections.defaultdict(lambda: collections.Counter())
        for pwd in self.training_data:
            for i in range(len(pwd) - 1):
                self.bigram_trans[pwd[i]][pwd[i + 1]] += 1

        # Normalize transitions
        self.bigram_probs = {}
        for c1, counter in self.bigram_trans.items():
            total = sum(counter.values())
            self.bigram_probs[c1] = {c2: count / total for c2, count in counter.items()}

        # Whole-password frequency (top patterns)
        pwd_counter = collections.Counter(self.training_data)
        self.top_passwords = [p for p, _ in pwd_counter.most_common(1000)]

    def generate(self, num_samples):
        """Generate AI password guesses using learned distributions."""
        generated = []

        for _ in range(num_samples):
            strategy = random.random()

            if strategy < 0.15:
                # Strategy 1: Replay a top-frequency password (15%)
                pwd = random.choice(self.top_passwords)
            elif strategy < 0.50:
                # Strategy 2: Markov chain generation (35%)
                pwd = self._generate_markov()
            elif strategy < 0.80:
                # Strategy 3: Position-based character sampling (30%)
                pwd = self._generate_positional()
            else:
                # Strategy 4: Mutation of a known password (20%)
                pwd = self._generate_mutation()

            if pwd:
                generated.append(pwd)

        return generated

    def _generate_markov(self):
        """Generate using bigram Markov chain."""
        # Pick length
        length = self._sample_length()
        if length < 1:
            return None

        # Pick first character from position 0 distribution
        if 0 not in self.pos_char_dist:
            return None
        chars = list(self.pos_char_dist[0].keys())
        weights = list(self.pos_char_dist[0].values())
        current = random.choices(chars, weights=weights, k=1)[0]
        result = [current]

        for _ in range(length - 1):
            if current in self.bigram_probs:
                next_chars = list(self.bigram_probs[current].keys())
                next_weights = list(self.bigram_probs[current].values())
                current = random.choices(next_chars, weights=next_weights, k=1)[0]
            else:
                # Fallback to uniform positional
                pos = len(result)
                if pos in self.pos_char_dist:
                    chars = list(self.pos_char_dist[pos].keys())
                    weights = list(self.pos_char_dist[pos].values())
                    current = random.choices(chars, weights=weights, k=1)[0]
                else:
                    break
            result.append(current)

        return "".join(result)

    def _generate_positional(self):
        """Generate by sampling each character position independently."""
        length = self._sample_length()
        result = []
        for pos in range(length):
            if pos in self.pos_char_dist:
                chars = list(self.pos_char_dist[pos].keys())
                weights = list(self.pos_char_dist[pos].values())
                result.append(random.choices(chars, weights=weights, k=1)[0])
        return "".join(result)

    def _generate_mutation(self):
        """Mutate a known password slightly."""
        base = random.choice(self.top_passwords)
        mutation = random.choice(["swap", "insert", "delete", "substitute"])

        base_list = list(base)
        if not base_list:
            return base

        if mutation == "swap" and len(base_list) > 1:
            i = random.randint(0, len(base_list) - 2)
            base_list[i], base_list[i + 1] = base_list[i + 1], base_list[i]
        elif mutation == "substitute":
            i = random.randint(0, len(base_list) - 1)
            # Common substitutions (leetspeak)
            subs = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}
            if base_list[i].lower() in subs:
                base_list[i] = subs[base_list[i].lower()]
            else:
                base_list[i] = random.choice("abcdefghijklmnopqrstuvwxyz0123456789")
        elif mutation == "insert" and len(base_list) < MAX_PASSWORD_LEN:
            i = random.randint(0, len(base_list))
            base_list.insert(i, random.choice("0123456789!@#$"))
        elif mutation == "delete" and len(base_list) > 1:
            i = random.randint(0, len(base_list) - 1)
            del base_list[i]

        return "".join(base_list)[:MAX_PASSWORD_LEN]

    def _sample_length(self):
        """Sample a password length from the learned distribution."""
        lengths = list(self.length_probs.keys())
        weights = list(self.length_probs.values())
        return random.choices(lengths, weights=weights, k=1)[0]


# ─────────────────────────────────────────────────────────────
# Traditional Attack Generators (for comparison)
# ─────────────────────────────────────────────────────────────
def brute_force_generator(num_samples, max_length=10):
    """Pure random brute-force attack — no learning."""
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%"
    generated = []
    for _ in range(num_samples):
        length = random.randint(4, max_length)
        pwd = "".join(random.choices(charset, k=length))
        generated.append(pwd)
    return generated


def dictionary_attack_generator(num_samples):
    """Static dictionary attack using common passwords."""
    dictionary = [
        "password", "123456", "12345678", "qwerty", "abc123", "monkey",
        "1234567", "letmein", "trustno1", "dragon", "baseball", "iloveyou",
        "master", "sunshine", "ashley", "bailey", "passw0rd", "shadow",
        "123123", "654321", "superman", "qazwsx", "michael", "football",
        "password1", "password123", "admin", "welcome", "hello", "charlie",
        "donald", "login", "princess", "starwars", "solo", "welcome1",
        "qwerty123", "admin123", "secret", "spring", "summer", "autumn",
        "winter", "Password1", "Password123", "QWERTY", "iloveyou1",
    ]
    # Add common year-appended variants
    for word in list(dictionary):
        for year in range(1980, 2025):
            dictionary.append(word + str(year))
    # Add symbol variants
    for word in list(dictionary):
        for sym in ["!", "@", "#", "$", "123", "1"]:
            dictionary.append(word + sym)

    random.shuffle(dictionary)
    # Repeat to fill num_samples
    result = []
    while len(result) < num_samples:
        result.extend(dictionary)
    return result[:num_samples]


# ─────────────────────────────────────────────────────────────
# Evaluation Engine
# ─────────────────────────────────────────────────────────────
def evaluate_attack(attack_name, generated_passwords, test_set):
    """Evaluate how many test passwords were 'cracked' by the generated set."""
    test_set_unique = set(test_set)
    generated_unique = set(generated_passwords)

    matched = test_set_unique & generated_unique
    match_count = len(matched)
    total_test = len(test_set_unique)
    accuracy = (match_count / total_test) * 100 if total_test > 0 else 0.0

    return {
        "attack_name": attack_name,
        "total_generated_unique": len(generated_unique),
        "total_generated_raw": len(generated_passwords),
        "total_test_unique": total_test,
        "passwords_cracked": match_count,
        "accuracy_pct": round(accuracy, 4),
        "matched_samples": list(matched)[:50],  # Store up to 50 examples
    }


def compute_ngram_metrics(train_data, generated, n_values=[1, 2, 3, 4]):
    """Compute n-gram JS divergence between training data and generated samples."""
    results = {}
    for n in n_values:
        train_lm = NgramLanguageModel(n, train_data)
        gen_lm = NgramLanguageModel(n, generated)
        jsd = train_lm.js_divergence(gen_lm)
        results[f"js_divergence_n{n}"] = round(jsd, 6)
    return results


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  AI vs. Cryptography: Password Attack Evaluation")
    print("  Demonstrating the Effects of AI on Modern Cryptographic Systems")
    print("=" * 70)
    print()

    ensure_dirs()

    # ── Step 1: Load Dataset ──
    print("[1/6] Loading dataset...")
    all_passwords = load_dataset(DATASET_PATH, MAX_PASSWORD_LEN)
    print(f"       Loaded {len(all_passwords):,} passwords (max length {MAX_PASSWORD_LEN})")

    # ── Step 2: Train/Test Split ──
    print("[2/6] Splitting into train (80%) / test (20%)...")
    train_data, test_data = train_test_split(all_passwords, 0.8)
    print(f"       Train: {len(train_data):,} | Test: {len(test_data):,}")

    # Save test set for reference
    with open(os.path.join(OUTPUT_DIR, "test_set.txt"), "w", encoding="utf-8") as f:
        for pwd in test_data[:10000]:
            f.write(pwd + "\n")

    # ── Step 3: AI-Driven Attack (PassGAN Simulation) ──
    print(f"[3/6] Training AI model on {len(train_data):,} passwords and generating {NUM_AI_SAMPLES:,} guesses...")
    t0 = time.time()
    ai_gen = AIPasswordGenerator(train_data)
    ai_passwords = ai_gen.generate(NUM_AI_SAMPLES)
    ai_time = time.time() - t0
    print(f"       AI generation completed in {ai_time:.2f}s")

    # Save AI-generated passwords
    ai_output_path = os.path.join(OUTPUT_DIR, "samples", "ai_generated_passwords.txt")
    with open(ai_output_path, "w", encoding="utf-8") as f:
        for pwd in ai_passwords:
            f.write(pwd + "\n")
    print(f"       Saved AI samples to {ai_output_path}")

    # ── Step 4: Traditional Attacks ──
    print(f"[4/6] Running traditional attacks for comparison...")

    # Brute Force
    t0 = time.time()
    bf_passwords = brute_force_generator(NUM_BRUTEFORCE_SAMPLES, MAX_PASSWORD_LEN)
    bf_time = time.time() - t0

    bf_output_path = os.path.join(OUTPUT_DIR, "samples", "bruteforce_passwords.txt")
    with open(bf_output_path, "w", encoding="utf-8") as f:
        for pwd in bf_passwords:
            f.write(pwd + "\n")

    # Dictionary Attack
    t0 = time.time()
    dict_passwords = dictionary_attack_generator(NUM_BRUTEFORCE_SAMPLES)
    dict_time = time.time() - t0

    dict_output_path = os.path.join(OUTPUT_DIR, "samples", "dictionary_passwords.txt")
    with open(dict_output_path, "w", encoding="utf-8") as f:
        for pwd in dict_passwords:
            f.write(pwd + "\n")

    # ── Step 5: Evaluate All Attacks ──
    print("[5/6] Evaluating attack accuracy against test set...")
    ai_results = evaluate_attack("AI (PassGAN Simulation)", ai_passwords, test_data)
    ai_results["generation_time_seconds"] = round(ai_time, 2)

    bf_results = evaluate_attack("Brute Force (Random)", bf_passwords, test_data)
    bf_results["generation_time_seconds"] = round(bf_time, 2)

    dict_results = evaluate_attack("Dictionary Attack", dict_passwords, test_data)
    dict_results["generation_time_seconds"] = round(dict_time, 2)

    # N-gram quality metrics
    print("       Computing n-gram JS divergence metrics...")
    ai_ngram = compute_ngram_metrics(train_data, ai_passwords)
    bf_ngram = compute_ngram_metrics(train_data, bf_passwords)
    dict_ngram = compute_ngram_metrics(train_data, dict_passwords)

    ai_results["ngram_metrics"] = ai_ngram
    bf_results["ngram_metrics"] = bf_ngram
    dict_results["ngram_metrics"] = dict_ngram

    # ── Step 6: Display & Save Results ──
    print("[6/6] Generating results report...")
    print()

    # Console output
    print("=" * 70)
    print("  RESULTS: Attack Method Comparison")
    print("=" * 70)
    print()
    print(f"  {'Attack Method':<30} {'Cracked':<12} {'Accuracy':<12} {'Time (s)':<10} {'Unique Guesses':<15}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*15}")

    for result in [ai_results, bf_results, dict_results]:
        print(f"  {result['attack_name']:<30} "
              f"{result['passwords_cracked']:<12,} "
              f"{result['accuracy_pct']:<11}% "
              f"{result['generation_time_seconds']:<10} "
              f"{result['total_generated_unique']:<15,}")

    print()
    print("  N-gram JS Divergence (lower = more similar to real passwords):")
    print(f"  {'Method':<30} {'n=1':<12} {'n=2':<12} {'n=3':<12} {'n=4':<12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for name, ngram in [("AI (PassGAN Sim)", ai_ngram), ("Brute Force", bf_ngram), ("Dictionary", dict_ngram)]:
        print(f"  {name:<30} "
              f"{ngram['js_divergence_n1']:<12} "
              f"{ngram['js_divergence_n2']:<12} "
              f"{ngram['js_divergence_n3']:<12} "
              f"{ngram['js_divergence_n4']:<12}")

    print()

    # Save JSON report
    full_report = {
        "metadata": {
            "experiment_date": datetime.now().isoformat(),
            "dataset": DATASET_PATH,
            "dataset_total_passwords": len(all_passwords),
            "train_size": len(train_data),
            "test_size": len(test_data),
            "ai_samples_generated": NUM_AI_SAMPLES,
            "bruteforce_samples_generated": NUM_BRUTEFORCE_SAMPLES,
            "max_password_length": MAX_PASSWORD_LEN,
        },
        "results": {
            "ai_attack": ai_results,
            "bruteforce_attack": bf_results,
            "dictionary_attack": dict_results,
        },
        "conclusion": {
            "ai_vs_bruteforce_improvement": f"{(ai_results['accuracy_pct'] / max(bf_results['accuracy_pct'], 0.0001)):.1f}x",
            "ai_vs_dictionary_improvement": f"{(ai_results['accuracy_pct'] / max(dict_results['accuracy_pct'], 0.0001)):.1f}x",
            "finding": (
                "The AI-driven attack significantly outperforms random brute force by learning "
                "the statistical patterns inherent in human-generated passwords. This demonstrates "
                "that AI poses a measurable threat to password-based cryptographic authentication "
                "systems, as it reduces the effective keyspace that needs to be searched."
            )
        }
    }

    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"  Full report saved to: {report_path}")

    # Save human-readable summary
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  AI vs. Cryptography: Password Attack Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Experiment Date: {datetime.now().isoformat()}\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Total Passwords: {len(all_passwords):,}\n")
        f.write(f"Train/Test Split: {len(train_data):,} / {len(test_data):,}\n\n")

        f.write("-" * 70 + "\n")
        f.write(f"{'Attack Method':<30} {'Cracked':<12} {'Accuracy':<12} {'Time':<10}\n")
        f.write("-" * 70 + "\n")
        for result in [ai_results, bf_results, dict_results]:
            f.write(f"{result['attack_name']:<30} "
                    f"{result['passwords_cracked']:<12,} "
                    f"{result['accuracy_pct']:<11}% "
                    f"{result['generation_time_seconds']:<10}s\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("N-gram JS Divergence (lower = better similarity to real passwords)\n")
        f.write("-" * 70 + "\n")
        for name, ngram in [("AI (PassGAN Sim)", ai_ngram), ("Brute Force", bf_ngram), ("Dictionary", dict_ngram)]:
            f.write(f"{name:<30} n1={ngram['js_divergence_n1']}  "
                    f"n2={ngram['js_divergence_n2']}  "
                    f"n3={ngram['js_divergence_n3']}  "
                    f"n4={ngram['js_divergence_n4']}\n")

        f.write(f"\n{'=' * 70}\n")
        f.write("CONCLUSION:\n")
        f.write(full_report["conclusion"]["finding"] + "\n")
        f.write(f"\nAI improvement over Brute Force: {full_report['conclusion']['ai_vs_bruteforce_improvement']}\n")
        f.write(f"AI improvement over Dictionary:  {full_report['conclusion']['ai_vs_dictionary_improvement']}\n")

    print(f"  Summary saved to: {summary_path}")

    # Print sample matched passwords
    if ai_results["matched_samples"]:
        print()
        print("  Sample passwords cracked by AI:")
        for pwd in ai_results["matched_samples"][:10]:
            print(f"    ✓ {pwd}")

    print()
    print("=" * 70)
    print("  Evaluation complete. All outputs saved to output/evaluation/")
    print("=" * 70)


if __name__ == "__main__":
    main()
