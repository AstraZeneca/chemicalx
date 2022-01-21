"""Download and pre-process the DrugBank drug-drug interaction dataset."""

import json
import math
from random import Random

import click
import pandas as pd
from utils import get_features, get_index, get_samples, get_tdc, map_context


@click.command()
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed")
@click.option("--ratio", type=float, default=1.0, show_default=True, help="Negative sampling ratio")
def main(seed: int, ratio: float):
    """Download and pre-process the DrugBank DDI dataset."""
    rng = Random(seed)
    input_directory, output_directory = get_tdc("drugbank", "drugbankddi")

    positive_samples = pd.read_csv(
        input_directory.joinpath("drugbank.tab"),
        sep="\t",
        usecols=[0, 1, 2, 4, 5],
        names=["drug_1", "drug_2", "context", "drug_1_smiles", "drug_2_smiles"],
    )
    print("Number of positive samples:", positive_samples.shape[0])

    contexts = list(set(positive_samples["Y"].values.tolist()))
    print("Number of contexts:", len(contexts))

    # Index drugs' SMILES and drug-drug-context triples
    drugs_raw, big_map = get_index(positive_samples)

    drugs = list(drugs_raw)
    print("Number of drugs:", len(drugs))

    # Generate negative samples
    negative_samples = get_samples(
        rng=rng, n=int(math.ceil(ratio * positive_samples.shape[0])), drugs=drugs, contexts=contexts, big_map=big_map
    )

    labeled_triples = positive_samples[["drug_1", "drug_2", "context"]]
    labeled_triples["label"] = 1.0
    labeled_triples = pd.concat([labeled_triples, negative_samples])
    labeled_triples["context"] = labeled_triples["context"].map(lambda x: f"context_{x}")
    print("Number of total triples:", labeled_triples.shape)
    labeled_triples.to_csv(output_directory.joinpath("labeled_triples.csv"), index=False)

    drugs_raw.update(
        {
            "DB09323": "O.O.O.O.C(CNCC1=CC=CC=C1)NCC1=CC=CC=C1.[H][C@]12SC(C)(C)[C@@H](N1C(=O)[C@H]2NC(=O)CC1=CC=CC=C1)C(O)=O.[H][C@]12SC(C)(C)[C@@H](N1C(=O)[C@H]2NC(=O)CC1=CC=CC=C1)C(O)=O",  # noqa:E501
            "DB13450": "[O-]S(=O)(=O)C1=CC=CC=C1.[O-]S(=O)(=O)C1=CC=CC=C1.COC1=CC2=C(C=C1OC)[C@@H](CC1=CC(OC)=C(OC)C=C1)[N@@+](C)(CCC(=O)OCCCCCOC(=O)CC[N@@+]1(C)CCC3=C(C=C(OC)C(OC)=C3)[C@H]1CC1=CC(OC)=C(OC)C=C1)CC2",  # noqa:E501
            "DB09396": "O.OS(=O)(=O)C1=CC2=CC=CC=C2C=C1.CCC(=O)O[C@@](CC1=CC=CC=C1)([C@H](C)CN(C)C)C1=CC=CC=C1",
            "DB09162": "[Fe+3].OC(CC([O-])=O)(CC([O-])=O)C([O-])=O",
            "DB11106": "CC(C)(N)CO.CN1C2=C(NC(Br)=N2)C(=O)N(C)C1=O",
            "DB11630": "C1CC2=NC1=C(C3=CC=C(N3)C(=C4C=CC(=N4)C(=C5C=CC(=C2C6=CC(=CC=C6)O)N5)C7=CC(=CC=C7)O)C8=CC(=CC=C8)O)C9=CC(=CC=C9)O",  # noqa:E501
            "DB00958": "C1CC(C1)(C(=O)O)C(=O)O.[NH2-].[NH2-].[Pt+2]",
            "DB00526": "C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]",
            "DB13145": "C(C(=O)O)O.[NH2-].[NH2-].[Pt+2]",
            "DB00515": "N.N.Cl[Pt]Cl",
        }
    )

    # Generate drugs file
    drug_set = {drug: {"smiles": smiles, "features": get_features(smiles)} for drug, smiles in drugs_raw.items()}
    with output_directory.joinpath("drug_set.json").open("w") as file:
        json.dump(drug_set, file)

    # Generate contexts file
    context_count = len(contexts)
    context_set = {context: map_context(i, context_count) for i, context in enumerate(contexts)}
    with output_directory.joinpath("context_set.json").open("w") as file:
        json.dump(context_set, file)


if __name__ == "__main__":
    main()
