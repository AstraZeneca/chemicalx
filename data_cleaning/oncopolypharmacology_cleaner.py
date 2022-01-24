"""Download and pre-process the OncoPolyPharmacology drug-drug synergy dataset."""

from itertools import chain

import click
import pandas as pd

from utils import get_tdc_synergy, write_contexts, write_drugs, write_triples


@click.command()
def main():
    """Download and pre-process the OncoPolyPharmacology synergy dataset."""
    input_directory, output_directory = get_tdc_synergy("OncoPolyPharmacology")
    df = pd.read_pickle(input_directory.joinpath("oncopolypharmacology.pkl"))
    # columns are Drug 1, Drug 2, cell line ID, cell line features (array), drug 1 smiles, drug 2 smiles

    drugs = dict(chain(
        df[["Drug1_ID", "Drug1"]].values,
        df[["Drug2_ID", "Drug2"]].values,
    ))
    write_drugs(drugs, output_directory)

    contexts = {key: values.round(4).tolist() for key, values in df[["Cell_Line_ID", "Cell_Line"]].values}
    write_contexts(contexts, output_directory)

    triples_df = df[["Drug1_ID", "Drug2_ID", "Cell_Line_ID", "Y"]].rename(columns={
        "Drug1_ID": "drug_1", "Drug2_ID": "drug_2", "Cell_Line_ID": "context",
        "Y": "label",
    })
    write_triples(triples_df, output_directory)


if __name__ == "__main__":
    main()
