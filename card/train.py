"""Reproducible training CLI for the card tier migration model."""

from __future__ import annotations

import argparse

from card.data import generate_portfolio
from card.modeling import train_migration_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Amex Premium Card Tier Migration & Margin Optimization model"
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic portfolio data before training.",
    )
    parser.add_argument(
        "--n-customers",
        type=int,
        default=25000,
        help="Number of synthetic customers to generate.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation (faster CI-style run).",
    )
    args = parser.parse_args()

    if args.generate_data:
        print(f"Generating synthetic portfolio ({args.n_customers:,} customers)...")
        generate_portfolio(n_customers=args.n_customers, random_state=42)
        print("Data generated.")

    print("Training migration & margin model...")
    train_migration_model(cv_splits=args.cv_splits, no_figures=args.no_figures)
    print("Training complete.")


if __name__ == "__main__":
    main()
