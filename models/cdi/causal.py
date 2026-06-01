import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSCM(nn.Module):
    """
    Structural Causal Model (SCM) module for CDI.
    Decouples visual representations into category-relevant (causal) 
    and category-agnostic (stylistic) features.
    """

    def __init__(self, feature_dim, z_dim=64, class_dim=32, style_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.z_dim = z_dim
        self.class_dim = class_dim
        self.style_dim = style_dim

        # Encoder: Decompose features into latent space
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, z_dim),
            nn.LayerNorm(z_dim)
        )

        # Causal content encoder (C) - Extracts essential category features
        self.content_encoder = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.LayerNorm(z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, class_dim)
        )

        # Style encoder (S) - Extracts stylistic features (background, illumination, etc.)
        self.style_encoder = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.LayerNorm(z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, style_dim)
        )

        # Decoder: Reconstruct features from causal and stylistic components
        self.decoder = nn.Sequential(
            nn.Linear(class_dim + style_dim, z_dim),
            nn.LayerNorm(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, feature_dim)
        )

        # MLP for structural residual intervention
        self.intervention_mlp = nn.Sequential(
            nn.Linear(class_dim, class_dim),
            nn.ReLU(),
            nn.Linear(class_dim, class_dim)
        )

        # Class projector to map causal features back to the original feature space
        self.class_projector = nn.Sequential(
            nn.Linear(class_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def encode(self, x):
        """Encode features into latent causal and stylistic representations."""
        z = self.encoder(x)
        c = self.content_encoder(z)  # Causal content
        s = self.style_encoder(z)    # Stylistic content
        return c, s

    def decode(self, c, s):
        """Decode representations back to the original feature space."""
        z_combined = torch.cat([c, s], dim=1)
        return self.decoder(z_combined)

    def intervene(self, c, strength=1.0):
        """Apply intervention to causal content features to reinforce category information."""
        c_intervened = c + strength * self.intervention_mlp(c)
        return c_intervened

    def counterfactual(self, x, y, mode='adaptive_noise', alpha=0.8):
        """
        Enhanced counterfactual generation: Apply diverse interventions to stylistic features.

        Args:
            x: Original features
            y: Labels
            mode: Intervention mode - 'adaptive_noise', 'masking', 'style_mixing', 'feature_corruption', 'domain_shift'
            alpha: Intervention strength parameter
        """
        # Encode to get causal and style features
        c_x, s_x = self.encode(x)
        batch_size = x.size(0)

        # Apply intervention based on the specified mode
        if mode == 'adaptive_noise':
            # Adaptive noise: add noise based on feature variance to simulate style shifts of novel classes
            s_std = torch.std(s_x, dim=0, keepdim=True)
            noise = torch.randn_like(s_x) * s_std * alpha
            s_cf = s_x + noise

        elif mode == 'masking':
            # Feature masking: adaptively mask style features to simulate feature absence
            feature_dim = s_x.size(1)

            # Calculate feature importance
            with torch.no_grad():
                # Estimate feature importance using gradient approximation
                s_perturbed = s_x.clone().detach().requires_grad_(True)
                x_recon = self.decode(c_x, s_perturbed)
                importance = torch.zeros_like(s_x)

                # Calculate importance for each feature dimension
                for i in range(feature_dim):
                    if i % 10 == 0:  # Sample every 10 dimensions for efficiency
                        s_temp = s_perturbed.clone()
                        s_temp[:, i] = 0  # Mask this dimension
                        x_temp = self.decode(c_x, s_temp)
                        diff = torch.norm(x_temp - x_recon, dim=1)
                        importance[:, i] = diff

            # Normalize importance scores
            importance = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)

            # Adaptive masking based on importance
            s_cf = s_x.clone()
            for i in range(batch_size):
                # Features with lower importance are more likely to be masked
                mask_prob = torch.sigmoid(5.0 * (importance[i] - importance[i].mean()))
                mask = torch.bernoulli(mask_prob)

                # Compensate masked features to maintain overall feature strength
                masked_sum = (s_x[i] * (1 - mask)).sum()
                original_sum = s_x[i].sum()
                if original_sum != 0:
                    scale = original_sum / (masked_sum + 1e-8)
                    s_cf[i] = s_x[i] * mask * scale
                else:
                    s_cf[i] = s_x[i] * mask

        elif mode == 'style_mixing':
            # Style mixing: mix with style features of other samples
            perm_idx = torch.randperm(batch_size)
            s_other = s_x[perm_idx]
            s_cf = alpha * s_x + (1 - alpha) * s_other

        elif mode == 'feature_corruption':
            # Feature corruption: simulate feature degradation
            corruption = torch.randn_like(s_x) * torch.mean(torch.abs(s_x), dim=1, keepdim=True) * alpha
            dropout_mask = torch.bernoulli(torch.ones_like(s_x) * 0.7)
            s_cf = s_x * dropout_mask + corruption

        elif mode == 'domain_shift':
            # Distribution Shift Counterfactual Intervention (DSCI)
            # Calculate global style statistics
            s_mean = torch.mean(s_x, dim=0, keepdim=True)
            s_centered = s_x - s_mean

            # Use SVD to decompose style space
            try:
                U, S, V = torch.svd(s_centered)
                # Select top-k principal components
                k = min(5, V.shape[1])
                principal_components = V[:, :k]
                singular_values = S[:k]
            except:
                # Fallback: eigendecomposition of covariance matrix (more stable for small matrices)
                k = min(5, s_x.size(1))
                # Calculate covariance matrix
                cov = torch.mm(s_centered.t(), s_centered) / (s_centered.size(0) - 1 + 1e-8)
                # Add slight regularization for numerical stability
                cov = cov + torch.eye(cov.size(0), device=cov.device) * 1e-6

                try:
                    # Use eigh (optimized for symmetric matrices)
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                    # Sort eigenvalues in descending order
                    idx = torch.argsort(eigenvalues, descending=True)
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]

                    principal_components = eigenvectors[:, :k]
                    # Convert eigenvalues to singular values
                    singular_values = torch.sqrt(torch.clamp(eigenvalues[:k] * (s_centered.size(0) - 1), min=1e-8))
                except:
                    # Final fallback: feature selection based on variance
                    feature_std = torch.std(s_x, dim=0)
                    importance_idx = torch.argsort(feature_std, descending=True)

                    principal_components = torch.eye(s_x.size(1), device=s_x.device)[:, importance_idx[:k]]
                    singular_values = feature_std[importance_idx[:k]] + 1e-8

            # Generate adaptive domain shift for each sample
            s_cf = s_x.clone()

            # Calculate class-specific shift directions
            class_shifts = {}
            for label in torch.unique(y):
                # Randomly generate a consistent shift direction for each class
                random_weights = torch.randn(k, device=s_x.device)
                # Ensure shift strength is proportional to feature norm
                shift_direction = torch.mm(random_weights.unsqueeze(0) * singular_values, principal_components.t())
                class_shifts[label.item()] = shift_direction

            # Apply class-specific domain shift
            for i in range(batch_size):
                label = y[i].item()
                shift_direction = class_shifts[label]

                # Calculate shift magnitude, proportional to sample feature norm
                shift_magnitude = torch.norm(s_x[i]) * alpha
                domain_shift = shift_direction * shift_magnitude / (torch.norm(shift_direction) + 1e-8)

                # Apply shift
                s_cf[i] = s_x[i] + domain_shift.squeeze(0)

        else:  
            # Default: multi-mode hybrid
            # Combine multiple intervention methods
            modes = ['adaptive_noise', 'masking', 'style_mixing', 'feature_corruption']
            mode_weights = F.softmax(torch.randn(len(modes)), dim=0)

            s_cf = s_x.clone()

            # Add adaptive noise
            noise_weight = mode_weights[0]
            s_std = torch.std(s_x, dim=0, keepdim=True)
            noise = torch.randn_like(s_x) * s_std * alpha * noise_weight
            s_cf = s_cf + noise

            # Apply feature masking
            mask_weight = mode_weights[1]
            mask = torch.bernoulli(torch.ones_like(s_x) * (1 - alpha * mask_weight))
            s_cf = s_cf * mask

            # Style mixing
            mix_weight = mode_weights[2]
            perm_idx = torch.randperm(batch_size)
            s_other = s_x[perm_idx]
            s_cf = s_cf * (1 - mix_weight) + s_other * mix_weight * alpha

            # Feature corruption
            corrupt_weight = mode_weights[3]
            corruption = torch.randn_like(s_x) * torch.mean(torch.abs(s_x), dim=1,
                                                            keepdim=True) * alpha * corrupt_weight
            s_cf = s_cf + corruption

        # Generate counterfactual samples using original causal features and intervened style features
        x_cf = self.decode(c_x, s_cf)

        return x_cf, c_x, s_cf

    def forward(self, x, intervention_strength=0.5, training=True):
        """Forward pass, including encode-intervene-decode process."""
        # Encode
        c, s = self.encode(x)

        # Intervene on causal content
        if training:
            c = self.intervene(c, intervention_strength)

        # Reconstruct features
        x_recon = self.decode(c, s)

        # Project causal features to the original feature space
        c_projected = self.class_projector(c)

        # Return features
        return {
            'class_features': c,
            'class_features_projected': c_projected,
            'style_features': s,
            'reconstructed_features': x_recon,
            'original_features': x
        }
