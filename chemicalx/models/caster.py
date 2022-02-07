"""An implementation of the CASTER model."""

from typing import Tuple

import torch

from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "CASTER",
]


class CASTER(Model):
    """An implementation of the CASTER model from [huang2020]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/17

    .. [huang2020] Huang, K., *et al.* (2020). `CASTER: Predicting drug interactions
       with chemical substructure representation <https://doi.org/10.1609/aaai.v34i01.5412>`_.
       *AAAI 2020 - 34th AAAI Conference on Artificial Intelligence*, 702–709.
    """

    def __init__(
        self,
        *,
        drug_channels: int,
        encoder_hidden_channels: int = 32,
        encoder_output_channels: int = 32,
        decoder_hidden_channels: int = 32,
        hidden_channels: int = 32,
        out_hidden_channels: int = 32,
        out_channels: int = 1,
        lambda3: float = 1e-5,
        magnifying_factor: int = 100,
    ):
        """Instantiate the CASTER model.

        :param drug_channels: The number of drug features (recognised frequent substructures).
            The original implementation recognised 1722 basis substructures in the BIOSNAP experiment.
        :param encoder_hidden_channels: The number of hidden layer neurons in the encoder module.
        :param encoder_output_channels: The number of output layer neurons in the encoder module.
        :param decoder_hidden_channels: The number of hidden layer neurons in the decoder module.
        :param hidden_channels: The number of hidden layer neurons in the predictor module.
        :param out_hidden_channels: The last hidden layer channels before output.
        :param out_channels: The number of output channels.
        :param lambda3: regularisation coefficient in the dictionary encoder module.
        :param magnifying_factor: The magnifying factor coefficient applied to the predictor module input.
        """
        super().__init__()
        self.lambda3 = lambda3
        self.magnifying_factor = magnifying_factor
        self.drug_channels = drug_channels

        # encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.drug_channels, encoder_hidden_channels),
            torch.nn.ReLU(True),
            torch.nn.Linear(encoder_hidden_channels, encoder_output_channels),
        )

        # decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_channels, decoder_hidden_channels),
            torch.nn.ReLU(True),
            torch.nn.Linear(decoder_hidden_channels, drug_channels),
        )

        # predictor: eight layer NN
        predictor_layers = []
        predictor_layers.append(torch.nn.Linear(self.drug_channels, hidden_channels))
        predictor_layers.append(torch.nn.ReLU(True))
        for i in range(1, 6):
            predictor_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            if i < 5:
                predictor_layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            else:
                predictor_layers.append(torch.nn.Linear(hidden_channels, out_hidden_channels))
            predictor_layers.append(torch.nn.ReLU(True))
        predictor_layers.append(torch.nn.Linear(out_hidden_channels, out_channels))
        predictor_layers.append(torch.nn.Sigmoid())
        self.predictor = torch.nn.Sequential(*predictor_layers)

    def unpack(self, batch: DrugPairBatch) -> Tuple[torch.FloatTensor]:
        """Return the "functional representation" of drug pairs, as defined in the original implementation.

        :param batch: batch of drug pairs
        :return: each pair is represented as a single vector with x^i = 1 if either x_1^i >= 1 or x_2^i >= 1
        """
        pair_representation = (torch.maximum(batch.drug_features_left, batch.drug_features_right) >= 1.0).float()
        return (pair_representation,)

    def dictionary_encoder(
        self, drug_pair_features_latent: torch.FloatTensor, dictionary_features_latent: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Perform a forward pass of the dictionary encoder submodule.

        :param drug_pair_features_latent: encoder output for the input drug_pair_features
            (batch_size x encoder_output_channels)
        :param dictionary_features_latent: projection of the drug_pair_features using the dictionary basis
            (encoder_output_channels x drug_channels)
        :return: sparse code X_o: (batch_size x drug_channels)
        """
        dict_feat_squared = torch.matmul(dictionary_features_latent, dictionary_features_latent.transpose(2, 1))
        dict_feat_squared_inv = torch.inverse(dict_feat_squared + self.lambda3 * (torch.eye(self.drug_channels)))
        dict_feat_closed_form = torch.matmul(dict_feat_squared_inv, dictionary_features_latent)
        r = drug_pair_features_latent[:, None, :].matmul(dict_feat_closed_form.transpose(2, 1)).squeeze(1)
        return r

    def forward(self, drug_pair_features: torch.FloatTensor) -> Tuple[torch.FloatTensor, ...]:
        """Run a forward pass of the CASTER model.

        :param drug_pair_features: functional representation of each drug pair (see unpack method)
        :return: (Tuple[torch.FloatTensor): a tuple of tensors including:
                prediction_scores: predicted target scores for each drug pair
                reconstructed: input drug pair vectors reconstructed by the encoder-decoder chain
                dictionary_encoded: drug pair features encoded by the dictionary encoder submodule
                dictionary_features_latent: projection of the encoded drug pair features using the dictionary basis
                drug_pair_features_latent: encoder output for the input drug_pair_features
                drug_pair_features: a copy of the input unpacked drug_pair_features (needed for loss calculation)
        """
        drug_pair_features_latent = self.encoder(drug_pair_features)
        dictionary_features_latent = self.encoder(torch.eye(self.drug_channels))
        dictionary_features_latent = dictionary_features_latent.mul(drug_pair_features[:, :, None])
        drug_pair_features_reconstructed = self.decoder(drug_pair_features_latent)
        reconstructed = torch.sigmoid(drug_pair_features_reconstructed)
        dictionary_encoded = self.dictionary_encoder(drug_pair_features_latent, dictionary_features_latent)
        prediction_scores = self.predictor(self.magnifying_factor * dictionary_encoded)

        return (
            prediction_scores,
            reconstructed,
            dictionary_encoded,
            dictionary_features_latent,
            drug_pair_features_latent,
            drug_pair_features,
        )
