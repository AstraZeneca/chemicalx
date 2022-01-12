class DrugPairBatch:
    """
    A data class to store a labeled drug pair batch.

    Args:
        identifiers (pd.DataFrame or None): A dataframe with drug pair, context and label columns.
        drug_features_left (torch.FloatTensor or None): A matrix of molecular features for the left hand drugs.
        drug_molecules_left (torchdrug.PackedGraph or None): Packed molecules for the left hand drugs.
        drug_features_right (torch.FloatTensor or None): A matrix of molecular features for the right hand drugs.
        drug_molecules_right (torchdrug.PackedGraph or None): Packed molecules for the right hand drugs.
        context_features (torch.FloatTensor or None): A matrix of biological/chemical context features.
        labels (torch.FloatTensor or None): A vector of drug pair labels.
    """

    def __init__(
        self,
        identifiers=None,
        drug_features_left=None,
        drug_molecules_left=None,
        drug_features_right=None,
        drug_molecules_right=None,
        context_features=None,
        labels=None,
    ):
        self.identifiers = identifiers
        self.drug_features_left = drug_features_left
        self.drug_molecules_left = drug_molecules_left
        self.drug_features_right = drug_features_right
        self.drug_molecules_right = drug_molecules_right
        self.context_features = context_features
        self.labels = labels
