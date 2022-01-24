"""Download and pre-process the TWOSIDES drug-drug interaction dataset."""

import math
import typing
from collections import Counter
from random import Random

import click
import pandas as pd
from tabulate import tabulate
from utils import get_index, get_samples, get_tdc, write_artifacts


@click.command()
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed")
@click.option("--ratio", type=float, default=1.0, show_default=True, help="Negative sampling ratio")
@click.option("--top", type=int, default=10, show_default=True, help="Keep top most common side effects")
def main(seed: int, ratio: float, top: int):
    """Download and pre-process the TWOSIDES dataset."""
    rng = Random(seed)
    input_directory, output_directory = get_tdc("TWOSIDES", "twosides")

    positive_samples = pd.read_csv(
        input_directory.joinpath("twosides.csv"),
        sep=",",
        header=0,
        usecols=[0, 1, 3, 4, 5],
        names=["drug_1", "drug_2", "context", "drug_1_smiles", "drug_2_smiles"],
    )
    print("Number of positive samples:", positive_samples.shape[0])

    context_counts: typing.Counter[str] = Counter(positive_samples["context"].values.tolist())
    contexts = sorted(key for key, _ in context_counts.most_common(top))
    print(tabulate(context_counts.most_common(top), headers=["context", "count"]))

    positive_samples = positive_samples[positive_samples["context"].isin(set(contexts))]
    print(positive_samples.shape)

    drugs_raw, big_map = get_index(positive_samples)

    drugs = list(drugs_raw.keys())
    print("Number of drugs:", len(drugs))

    # Generate negative samples
    negative_samples = get_samples(
        rng=rng, n=int(math.ceil(ratio * positive_samples.shape[0])), drugs=drugs, contexts=contexts, big_map=big_map
    )

    labeled_triples = positive_samples[["drug_1", "drug_2", "context"]]
    labeled_triples["label"] = 1.0
    labeled_triples = pd.concat([labeled_triples, negative_samples])
    print("Number of total triples:", labeled_triples.shape)
    labeled_triples.to_csv(output_directory.joinpath("labeled_triples.csv"), index=False)

    write_artifacts(output_directory=output_directory, drugs_raw=drugs_raw, contexts=contexts)


if __name__ == "__main__":
    main()
