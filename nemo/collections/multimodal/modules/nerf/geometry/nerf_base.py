# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import Enum
from typing import Optional, Tuple

import mcubes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from nemo.collections.multimodal.modules.nerf.utils.activation import trunc_exp


class DensityActivationEnum(str, Enum):
    EXP = "exp"
    SOFTPLUS = "softplus"


class NormalTypeEnum(str, Enum):
    AUTOGRAD = "autograd"
    FORWARD_FINITE_DIFFERENCE = "forward_finite_difference"
    BACKWARD_FINITE_DIFFERENCE = "backward_finite_difference"
    CENTRAL_FINITE_DIFFERENCE = "central_finite_difference"


# TODO(ahmadki): make abstract
class NeRFBase(nn.Module):
    """
    A base class for Neural Radiance Fields (NeRF) models.

    Args:
        num_input_dims (int): Number of input dimensions.
        bound (torch.Tensor): The bounding box tensor.
        density_activation (DensityActivationEnum): Activation function for density.
        blob_radius (float): Radius for the blob.
        blob_density (float): Density for the blob.
        normal_type (Optional[NormalTypeEnum]): Method to compute normals.
    """

    def __init__(
        self,
        num_input_dims: int,
        bound: torch.Tensor,
        density_activation: DensityActivationEnum,
        blob_radius: float,
        blob_density: float,
        normal_type: Optional[NormalTypeEnum] = NormalTypeEnum.CENTRAL_FINITE_DIFFERENCE,
    ) -> None:
        super().__init__()
        self.num_input_dims = num_input_dims
        self.bound = bound
        self.density_activation = density_activation
        self.blob_radius = blob_radius
        self.blob_density = blob_density
        self.normal_type = normal_type

    def encode(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode 3D positions. To be implemented by subclasses."""
        raise NotImplementedError

    def sigma_net(self, positions_encoding: torch.Tensor) -> torch.Tensor:
        """Calculate sigma (density). To be implemented by subclasses."""
        raise NotImplementedError

    def features_net(self, positions_encoding: torch.Tensor) -> torch.Tensor:
        """Calculate features. To be implemented by subclasses."""
        raise NotImplementedError

    def forward(
        self, positions: torch.Tensor, return_normal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the NeRF model.

        Args:
            positions (torch.Tensor): The positions.
            return_normal (bool): Flag to indicate whether to return normals or not.

        Returns:
            Tuple containing density, features, and possibly normals.
        """

        if return_normal:
            if self.normal_type == NormalTypeEnum.AUTOGRAD:
                with torch.enable_grad():
                    positions.requires_grad_(True)
                    sigma, features = self.forward_density_features(positions)
                    normal = -torch.autograd.grad(torch.sum(sigma), positions, create_graph=True)[0]  # [N, D]
            elif self.normal_type in [
                NormalTypeEnum.CENTRAL_FINITE_DIFFERENCE,
                NormalTypeEnum.FORWARD_FINITE_DIFFERENCE,
                NormalTypeEnum.BACKWARD_FINITE_DIFFERENCE,
            ]:
                sigma, features = self.forward_density_features(positions)
                normal = self.normal_finite_differences(positions)
            else:
                raise NotImplementedError("Invalid normal type.")

            normal = F.normalize(normal)
            normal = torch.nan_to_num(normal)
        else:
            sigma, features = self.forward_density_features(positions)
            normal = None

        return sigma, features, normal

    def forward_density_features(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate both density and features based on the input positions.

        This function takes into account edge cases like empty input tensors and calculates
        the density and features accordingly. See GitHub issues for details:
        - https://github.com/KAIR-BAIR/nerfacc/issues/207#issuecomment-1653621720
        - https://github.com/ashawkey/torch-ngp/issues/176

        Args:
            positions (torch.Tensor): Input positions tensor with shape [B*N, D].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing density and features tensors.
        """

        # Handle empty positions
        if positions.shape[0] == 0:
            sigma = torch.zeros(0, device=positions.device)
            features = torch.zeros(0, self.num_input_dims, device=positions.device)
            return sigma, features

        # Encode positions
        positions_encoding = self.encode(positions)

        # Compute density
        density = self.forward_density(positions, positions_encoding)

        # Compute features
        features = self.forward_features(positions, positions_encoding)

        return density, features

    def forward_density(
        self, positions: torch.Tensor, positions_encoding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate the density based on the input positions and their encoding.

        Args:
            positions (torch.Tensor): Input positions tensor with shape [B*N, D].
            positions_encoding (Optional[torch.Tensor]): Optional encoded positions.
                Will be computed from `positions` if not provided.

        Returns:
            torch.Tensor: Density tensor.
        """

        # Handle empty positions
        if positions.shape[0] == 0:
            sigma = torch.zeros(0, device=positions.device)
            return sigma

        # Compute encoded positions if not provided
        if positions_encoding is None:
            positions_encoding = self.encode(positions)

        # Compute sigma using the neural network
        sigma = self.sigma_net(positions_encoding)

        # Compute density using activation function
        if self.density_activation == DensityActivationEnum.EXP:
            density = trunc_exp(sigma + self.density_blob(positions))
        elif self.density_activation == DensityActivationEnum.SOFTPLUS:
            density = F.softplus(sigma + self.density_blob(positions))
        else:
            raise NotImplementedError("Invalid density activation.")

        return density

    def forward_features(
        self, positions: torch.Tensor, positions_encoding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the features based on the input positions and their encoding.

        Args:
            positions (torch.Tensor): Input positions tensor with shape [B*N, D].
            positions_encoding (Optional[torch.Tensor]): Optional encoded positions.
                Will be computed from `positions` if not provided.

        Returns:
            torch.Tensor: Features tensor with shape [B*N, num_features_dims].
        """

        # Handle empty positions
        if positions.shape[0] == 0:
            features = torch.zeros(0, self.num_features_dims, device=positions.device)
            return features

        # Compute encoded positions if not provided
        if positions_encoding is None:
            positions_encoding = self.encode(positions)

        # Compute features using the neural network
        features = self.features_net(positions_encoding)

        # Apply the sigmoid activation function to the features
        features = torch.sigmoid(features)

        return features

    @torch.no_grad()
    def density_blob(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute the density blob for the given positions.

        This method computes a density blob for each position in the tensor. It is
        used to add a density value based on the distance of each position from the origin.

        Args:
            positions (torch.Tensor): Input positions tensor with shape [B*N, D].

        Returns:
            torch.Tensor: Density blob tensor with shape [B*N, 1].
        """

        # Compute the squared distance for each position
        d = (positions ** 2).sum(-1)

        # Compute the density blob based on the activation function
        if self.density_activation == DensityActivationEnum.EXP:
            g = self.blob_density * torch.exp(-d / (2 * self.blob_radius ** 2))
        elif self.density_activation == DensityActivationEnum.SOFTPLUS:
            g = self.blob_density * (1 - torch.sqrt(d) / self.blob_radius)
        else:
            raise NotImplementedError("Invalid density activation.")

        return g

    def normal_finite_differences(self, positions: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
        """
        Calculate normals using finite differences.

        Args:
            positions (torch.Tensor): Input positions tensor with shape [B*N, D].
            eps (float): A small value for finite difference calculation. Default is 1e-2.

        Returns:
            torch.Tensor: Calculated normals tensor [B*N, D]
        """
        # Create perturbation tensor
        perturb = torch.eye(self.num_input_dims).to(positions.device).float() * eps  # Shape (D, D)

        # Expand dims for batched operation
        positions_expanded = positions[:, None, :]  # (B*N, 1, D)
        perturb_expanded = perturb[None, :, :]  # (1, D, D)

        # Compute perturbed points
        if self.normal_type == NormalTypeEnum.FORWARD_FINITE_DIFFERENCE:
            positions_perturbed = positions_expanded + perturb_expanded  # (B*N, D, D)
        elif self.normal_type == NormalTypeEnum.BACKWARD_FINITE_DIFFERENCE:
            positions_perturbed = positions_expanded - perturb_expanded  # (B*N, D, D)
        elif self.normal_type == NormalTypeEnum.CENTRAL_FINITE_DIFFERENCE:
            positions_perturbed_pos = positions_expanded + perturb_expanded  # (B*N, D, D)
            positions_perturbed_neg = positions_expanded - perturb_expanded  # (B*N, D, D)
            positions_perturbed = torch.cat([positions_perturbed_pos, positions_perturbed_neg], dim=1)  # (B*N, 2*D, D)

        # Reshape perturbed points for batched function call
        positions_perturbed_reshaped = positions_perturbed.view(-1, self.num_input_dims)  # (B*N * {D or 2*D}, D)

        # Evaluate function at perturbed points
        perturbed_sigma = self.forward_density(positions_perturbed_reshaped)  # (B*N * {D or 2*D}, 1)

        # Reshape function values
        if self.normal_type == NormalTypeEnum.CENTRAL_FINITE_DIFFERENCE:
            perturbed_sigma = perturbed_sigma.view(-1, 2 * self.num_input_dims)  # (B*N, 2*D)
            sigma_pos, sigma_neg = torch.chunk(perturbed_sigma, 2, dim=1)  # (B*N, D) each
            normal = 0.5 * (sigma_pos - sigma_neg) / eps  # (B*N, D)
        else:
            perturbed_sigma = perturbed_sigma.view(-1, self.num_input_dims)  # (B*N, D)
            sigma = self.forward_density(positions)  # (B*N,) # TODO(ahmadki): use the value from forward ?
            if self.normal_type == NormalTypeEnum.FORWARD_FINITE_DIFFERENCE:
                normal = (perturbed_sigma - sigma[:, None]) / eps  # (B*N, D)
            else:  # self.normal_type == BACKWARD_FINITE_DIFFERENCE
                normal = (sigma[:, None] - perturbed_sigma) / eps  # (B*N, D)

        return -normal

    # TODO(ahmadki): needs ar ework:
    # 1. texture/vertices are off-axis, needs a fix.
    # 2. device='cuda' is hardcoded
    # 3. DMTet needs to go through a different code path ? create a base volume nerf, and a base dmtet nerf class ?
    @torch.no_grad()
    def mesh(
        self, resolution: Optional[int] = 128, batch_size: int = 128, density_thresh: Optional[float] = None
    ) -> trimesh.base.Trimesh:
        """
        Generate a mesh from the nerf.

        Args:
            resolution (Optional[int]): Resolution of the mesh grid. Default is 128.
            batch_size (int): Batch size for the mesh generation. Default is 128.
            density_thresh (Optional[float]): Density threshold for the mesh generation. Default is None, will be calculated from mean density.

        Returns:
            trimesh.base.Trimesh: Mesh object.
        """
        # Generate a grid of 3D points
        x = np.linspace(-self.bound, self.bound, resolution)
        y = np.linspace(-self.bound, self.bound, resolution)
        z = np.linspace(-self.bound, self.bound, resolution)
        xx, yy, zz = np.meshgrid(x, y, z)

        grid = np.stack((xx, yy, zz), axis=-1)  # Shape (resolution, resolution, resolution, 3)
        torch_grid = torch.tensor(grid, dtype=torch.float32).reshape(-1, 3).to(device="cuda")

        def batch_process(fn, input, batch_size):
            num_points = input.shape[0]
            batches = [input[i : i + batch_size] for i in range(0, num_points, batch_size)]
            results = [fn(batch) for batch in batches]
            results = [result.detach().cpu().numpy() for result in results]
            return np.concatenate(results, axis=0)

        density = batch_process(fn=self.forward_density, input=torch_grid, batch_size=batch_size)
        density = density.reshape(resolution, resolution, resolution)

        # If not provided set density_thresh based on mean density
        if density_thresh is None:
            density_thresh = density[density > 1e-3].mean().item()

        # Apply Marching Cubes
        vertices, triangles = mcubes.marching_cubes(density, density_thresh)

        # Create a new Mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Basic mesh cleaning and optimization
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()
        mesh.remove_duplicate_faces()

        # Scale vertices back to [-self.bound, self.bound]
        scaled_vertices = -self.bound + (mesh.vertices / resolution) * 2 * self.bound
        mesh.vertices = scaled_vertices

        # Assigning color to vertices
        scaled_vertices_torch = torch.tensor(scaled_vertices, dtype=torch.float32).to(device="cuda")
        color = batch_process(fn=self.forward_features, input=scaled_vertices_torch, batch_size=batch_size)
        color = (color * 255).astype(np.uint8)
        mesh.visual.vertex_colors = color

        return mesh
