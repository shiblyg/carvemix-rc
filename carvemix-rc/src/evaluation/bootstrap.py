import argparse

import numpy as np
import pandas as pd


def paired_bootstrap(scores_a, scores_b, patients, n_iterations, seed):
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patients)
    differences = np.empty(n_iterations)
    for i in range(n_iterations):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        mask = np.isin(patients, sampled)
        differences[i] = scores_b[mask].mean() - scores_a[mask].mean()
    return differences


def run(scores_a_path, scores_b_path, patient_column, score_column,
        n_iterations, seed):
    a = pd.read_csv(scores_a_path)
    b = pd.read_csv(scores_b_path)
    differences = paired_bootstrap(
        a[score_column].to_numpy(), b[score_column].to_numpy(),
        a[patient_column].to_numpy(), n_iterations, seed)

    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
    p_value = 2 * min((differences <= 0).mean(), (differences >= 0).mean())
    print(f"Median A: {a[score_column].median():.3f}")
    print(f"Median B: {b[score_column].median():.3f}")
    print(f"Mean paired difference: {differences.mean():+.3f}")
    print(f"95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"p-value: {p_value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Patient-clustered paired bootstrap.")
    parser.add_argument("--scores-a", required=True)
    parser.add_argument("--scores-b", required=True)
    parser.add_argument("--patient-column", default="patient")
    parser.add_argument("--score-column", default="dsc")
    parser.add_argument("--n-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.scores_a, args.scores_b, args.patient_column, args.score_column,
        args.n_iterations, args.seed)


if __name__ == "__main__":
    main()
