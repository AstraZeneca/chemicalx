class DrugPairBatch:
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
