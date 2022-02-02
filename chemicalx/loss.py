"""Custom loss modules for chemicalx."""

from typing import Tuple

import torch
from torch.nn.modules.loss import _Loss

__all__ = [
    "CASTERSupervisedLoss",
]


class CASTERSupervisedLoss(_Loss):
    def __init__(
        self, recon_loss_coeff: float = 1e-1, proj_coeff: float = 1e-1, lambda1: float = 1e-2, lambda2: float = 1e-1
    ):
        """
        Custom loss function for the supervised learning stage of the CASTER algorithm.

        :param recon_loss_coeff: coefficient for the reconstruction loss
        :param proj_coeff: coefficient for the projection loss
        :param lambda1: regularization coefficient for the projection loss
        :param lambda2: regularization coefficient for the augmented projection loss
        """
        super().__init__(reduction="none")
        self.recon_loss_coeff = recon_loss_coeff
        self.proj_coeff = proj_coeff
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss = torch.nn.BCELoss()

    def forward(self, x: Tuple[torch.FloatTensor, ...], target: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass of the loss calculation for the supervised learning stage of the CASTER algorithm.

        :param x: a tuple of tensors returned by the model forward pass (see CASTER.forward() method)
        :param target: target labels
        :return: combined loss value
        """
        score, recon, code, dictionary_features_latent, drug_pair_features_latent, drug_pair_features = x
        batch_size, _ = drug_pair_features.shape
        loss_prediction = self.loss(score, target.float())
        loss_reconstruction = self.recon_loss_coeff * self.loss(recon, drug_pair_features)
        loss_projection = self.proj_coeff * (
            torch.norm(drug_pair_features_latent - torch.matmul(code, dictionary_features_latent))
            + self.lambda1 * torch.sum(torch.abs(code)) / batch_size
            + self.lambda2 * torch.norm(dictionary_features_latent, p="fro") / batch_size
        )
        loss = loss_prediction + loss_reconstruction + loss_projection
        return loss
