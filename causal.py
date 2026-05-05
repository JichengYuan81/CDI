import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSCM(nn.Module):
    def __init__(self, feature_dim, z_dim=64, class_dim=32, style_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.z_dim = z_dim
        self.class_dim = class_dim
        self.style_dim = style_dim

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, z_dim),
            nn.LayerNorm(z_dim)
        )

        self.content_encoder = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.LayerNorm(z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, class_dim)
        )

        self.style_encoder = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.LayerNorm(z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, style_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(class_dim + style_dim, z_dim),
            nn.LayerNorm(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, feature_dim)
        )

        self.intervention_mlp = nn.Sequential(
            nn.Linear(class_dim, class_dim),
            nn.ReLU(),
            nn.Linear(class_dim, class_dim)
        )

        self.class_projector = nn.Sequential(
            nn.Linear(class_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def encode(self, x):
        z = self.encoder(x)
        c = self.content_encoder(z)
        s = self.style_encoder(z)
        return c, s

    def decode(self, c, s):
        z_combined = torch.cat([c, s], dim=1)
        return self.decoder(z_combined)

    def intervene(self, c, strength=1.0):
        c_intervened = c + strength * self.intervention_mlp(c)
        return c_intervened

    def counterfactual(self, x, y, mode='adaptive_noise', alpha=0.8):
        Args:
            x: Original features
            y: Label
            mode: Intervention approaches - 'adaptive_noise', 'masking', 'style_mixing', 'feature_corruption'
            alpha: Mixed strength parameters
        """

        c_x, s_x = self.encode(x)
        batch_size = x.size(0)


        if mode == 'adaptive_noise':
            s_std = torch.std(s_x, dim=0, keepdim=True)
            noise = torch.randn_like(s_x) * s_std * alpha
            s_cf = s_x + noise

        elif mode == 'masking':
            # mask = torch.bernoulli(torch.ones_like(s_x) * (1 - alpha))
            # s_cf = s_x * mask
            feature_dim = s_x.size(1)

            with torch.no_grad():
                s_perturbed = s_x.clone().detach().requires_grad_(True)
                x_recon = self.decode(c_x, s_perturbed)
                importance = torch.zeros_like(s_x)

                for i in range(feature_dim):
                    if i % 10 == 0:
                        s_temp = s_perturbed.clone()
                        s_temp[:, i] = 0
                        x_temp = self.decode(c_x, s_temp)
                        diff = torch.norm(x_temp - x_recon, dim=1)
                        importance[:, i] = diff


            importance = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)


            s_cf = s_x.clone()
            for i in range(batch_size):
                mask_prob = torch.sigmoid(5.0 * (importance[i] - importance[i].mean()))
                mask = torch.bernoulli(mask_prob)

                masked_sum = (s_x[i] * (1 - mask)).sum()
                original_sum = s_x[i].sum()
                if original_sum != 0:
                    scale = original_sum / (masked_sum + 1e-8)
                    s_cf[i] = s_x[i] * mask * scale
                else:
                    s_cf[i] = s_x[i] * mask

        elif mode == 'style_mixing':
            perm_idx = torch.randperm(batch_size)
            s_other = s_x[perm_idx]
            s_cf = alpha * s_x + (1 - alpha) * s_other

        elif mode == 'feature_corruption':
            corruption = torch.randn_like(s_x) * torch.mean(torch.abs(s_x), dim=1, keepdim=True) * alpha
            dropout_mask = torch.bernoulli(torch.ones_like(s_x) * 0.7)
            s_cf = s_x * dropout_mask + corruption

        elif mode == 'domain_shift':
            s_mean = torch.mean(s_x, dim=0, keepdim=True)
            s_centered = s_x - s_mean

            try:
                U, S, V = torch.svd(s_centered)
                k = min(5, V.shape[1])
                principal_components = V[:, :k]
                singular_values = S[:k]
                # print(f"✓ SVD: k={k}, singular_values scale=[{S[0]:.4f}, {S[k - 1]:.4f}]")
            except:
                k = min(5, s_x.size(1))


                cov = torch.mm(s_centered.t(), s_centered) / (s_centered.size(0) - 1 + 1e-8)
                cov = cov + torch.eye(cov.size(0), device=cov.device) * 1e-6

                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

                    idx = torch.argsort(eigenvalues, descending=True)
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]

                    principal_components = eigenvectors[:, :k]
                    singular_values = torch.sqrt(torch.clamp(eigenvalues[:k] * (s_centered.size(0) - 1), min=1e-8))
                    print(f"✓ Covariance decomposition successful: k={k}, eigenvalues scale=[{eigenvalues[0]:.4f}, {eigenvalues[k - 1]:.4f}]")
                except:
                    feature_std = torch.std(s_x, dim=0)
                    importance_idx = torch.argsort(feature_std, descending=True)

                    principal_components = torch.eye(s_x.size(1), device=s_x.device)[:, importance_idx[:k]]
                    singular_values = feature_std[importance_idx[:k]] + 1e-8

            s_cf = s_x.clone()

            class_shifts = {}
            for label in torch.unique(y):
                random_weights = torch.randn(k, device=s_x.device)
                shift_direction = torch.mm(random_weights.unsqueeze(0) * singular_values, principal_components.t())
                class_shifts[label.item()] = shift_direction

            for i in range(batch_size):
                label = y[i].item()
                shift_direction = class_shifts[label]

                shift_magnitude = torch.norm(s_x[i]) * alpha
                domain_shift = shift_direction * shift_magnitude / (torch.norm(shift_direction) + 1e-8)

                s_cf[i] = s_x[i] + domain_shift.squeeze(0)

        else: 
            modes = ['adaptive_noise', 'masking', 'style_mixing', 'feature_corruption']
            mode_weights = F.softmax(torch.randn(len(modes)), dim=0)

            s_cf = s_x.clone()

            noise_weight = mode_weights[0]
            s_std = torch.std(s_x, dim=0, keepdim=True)
            noise = torch.randn_like(s_x) * s_std * alpha * noise_weight
            s_cf = s_cf + noise

            mask_weight = mode_weights[1]
            mask = torch.bernoulli(torch.ones_like(s_x) * (1 - alpha * mask_weight))
            s_cf = s_cf * mask

            mix_weight = mode_weights[2]
            perm_idx = torch.randperm(batch_size)
            s_other = s_x[perm_idx]
            s_cf = s_cf * (1 - mix_weight) + s_other * mix_weight * alpha

            corrupt_weight = mode_weights[3]
            corruption = torch.randn_like(s_x) * torch.mean(torch.abs(s_x), dim=1,
                                                            keepdim=True) * alpha * corrupt_weight
            s_cf = s_cf + corruption

        x_cf = self.decode(c_x, s_cf)

        return x_cf, c_x, s_cf

    def forward(self, x, intervention_strength=0.5, training=True):
        c, s = self.encode(x)

        if training:
            c = self.intervene(c, intervention_strength)

        x_recon = self.decode(c, s)

        c_projected = self.class_projector(c)

        return {
            'class_features': c,
            'class_features_projected': c_projected,
            'style_features': s,
            'reconstructed_features': x_recon,
            'original_features': x
        }
