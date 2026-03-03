import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ViTEncoder


class AnomExpert(nn.Module):
    """AnomExpert: Prototype-driven framework for prenatal ultrasound anomaly diagnosis.
    
    Core pipeline (matching paper Sec.2):
    1. Image embedding with ViT backbone
    2. Plane prototype learning (Sinkhorn assignment)
    3. Disease-aware sparse plane selection
    4. Case-level anomaly classification
    """
    def __init__(
        self,
        num_disease_categories: int,  # C in paper
        encoder_dim: int = 768,
        hidden_dim: int = 256,       # 256 as in paper Eq.(1)
        num_plane_protos: int = 30,  # K in paper (default 30)
        topk_planes: int = 4,        # top-k in paper Sec.2.2 (default 4)
        attn_temperature: float = 0.07,  # T in paper Eq.(8)
        sinkhorn_eps: float = 0.05,      # ε_s in paper Eq.(2)
        sinkhorn_iter: int = 3,
    ):
        super().__init__()

        # Core hyperparameters (matching paper notations)
        self.C = num_disease_categories  # Number of disease categories
        self.K = num_plane_protos        # Number of plane prototypes
        self.D = hidden_dim              # Embedding dimension
        self.topk_planes = topk_planes
        self.attn_T = attn_temperature
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iter = sinkhorn_iter

        # 1. Image encoder (ViT-small as default in paper)
        self.vit_backbone = ViTEncoder('tiny')
        self.projection_head = nn.Linear(encoder_dim, self.D)  # g(·) in paper Eq.(1)

        # 2. Plane prototype learning components
        self.plane_prototypes = nn.Parameter(torch.randn(self.K, self.D) * 0.02)  # P in paper Sec.2.1
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))        # τ in paper Eq.(3)

        # 3. Disease-aware plane selection components
        self.disease_queries = nn.Parameter(torch.randn(1, self.C, self.D) * 0.02)  # Q in paper Eq.(5)
        self.disease_bias_matrix = nn.Parameter(torch.zeros(self.C, self.K))        # B in paper Eq.(7)

        # 4. Classification head (concat [z_1,...,z_C] -> prediction)
        self.classifier_head = nn.Linear(self.C * self.D, self.C)


    def _process_var_len_seqs(self, img_embeddings: torch.Tensor, case_lengths: list) -> tuple:
        """Pad variable-length case sequences to fixed batch size (matching paper's case-level input).
        
        Args:
            img_embeddings: Encoded image embeddings (total_imgs, D)
            case_lengths: List of image counts per case (sum=total_imgs)
        Returns:
            batch_padded_emb: Padded embeddings (B, L_max, D)
            batch_pad_mask: Padding mask (B, L_max), True=PAD
        """
        max_len = max(case_lengths)
        device = img_embeddings.device

        batch_padded_emb, batch_pad_mask = [], []
        start_idx = 0

        for case_len in case_lengths:
            # Extract embeddings for current case
            case_emb = img_embeddings[start_idx: start_idx + case_len]
            
            # Initialize padding (constant value for PAD positions)
            pad_emb = torch.ones((max_len, self.D), device=device) * 0.1
            pad_mask = torch.ones((max_len,), dtype=torch.bool, device=device)

            # Fill valid embeddings and update mask
            pad_emb[:case_len] = case_emb
            pad_mask[:case_len] = False

            start_idx += case_len
            batch_padded_emb.append(pad_emb)
            batch_pad_mask.append(pad_mask)

        # Stack to batch tensors
        batch_padded_emb = torch.stack(batch_padded_emb)  # (B, L_max, D)
        batch_pad_mask = torch.stack(batch_pad_mask)      # (B, L_max)

        return batch_padded_emb, batch_pad_mask


    def forward(self, images: torch.Tensor, case_lengths: list) -> tuple:
        """Forward pass of AnomExpert (matching paper Sec.2).
        
        Args:
            images: Input ultrasound images (total_imgs, 3, 224, 224)
            case_lengths: Image counts per case (list[B])
        Returns:
            pred_logits: Case-level prediction logits (B, C)
            loss_proto: Prototype assignment loss (scalar)
            aux: Auxiliary outputs for visualization
        """
        # Step 1: Image embedding (paper Eq.(1))
        img_features = self.vit_backbone(images)                # (total_imgs, encoder_dim)
        img_embeddings = self.projection_head(img_features)     # (total_imgs, D)

        # Step 2: Pad to fixed batch shape (case-level processing)
        batch_padded_emb, batch_pad_mask = self._process_var_len_seqs(img_embeddings, case_lengths)
        B, L, D = batch_padded_emb.shape
        assert D == self.D, f"Embedding dim mismatch: {D} vs {self.D}"

        # Step 3: Plane prototype learning (paper Sec.2.1)
        loss_proto, assignment_matrix = self._plane_prototype_learning(batch_padded_emb, batch_pad_mask)

        # Step 4: Aggregate to plane representations (paper Eq.(4))
        plane_representations, cluster_valid_mask = self._aggregate_to_plane_representations(
            batch_padded_emb, batch_pad_mask, assignment_matrix
        )

        # Step 5: Disease-aware plane selection (paper Sec.2.2)
        disease_embeddings, sparse_attention = self._disease_aware_plane_selection(
            plane_representations, cluster_valid_mask
        )

        # Step 6: Case-level classification (paper Eq.(9))
        pred_logits = self.classifier_head(disease_embeddings.reshape(B, -1))

        # Auxiliary outputs for visualization (matching paper Fig.3)
        aux = {
            "prototype_assignment": assignment_matrix,  # (B, L, K)
            "plane_representations": plane_representations,  # (B, K, D)
            "disease_attention_weights": sparse_attention  # (B, C, K)
        }

        return pred_logits, loss_proto, aux


    def _plane_prototype_learning(self, batch_padded_emb: torch.Tensor, batch_pad_mask: torch.Tensor):
        """Plane prototype learning with Sinkhorn assignment (paper Sec.2.1).
        
        Args:
            batch_padded_emb: Padded image embeddings (B, L, D)
            batch_pad_mask: Padding mask (B, L), True=PAD
        Returns:
            loss_proto: Prototype assignment loss (paper Eq.(3))
            assignment_matrix: Sinkhorn assignment matrix (B, L, K)
        """
        B, L, D = batch_padded_emb.shape

        # Flatten and filter valid embeddings (exclude PAD)
        flat_emb = batch_padded_emb.reshape(-1, D)          # (B*L, D)
        flat_valid_mask = (~batch_pad_mask).reshape(-1)     # (B*L,)
        valid_emb = flat_emb[flat_valid_mask]               # (N, D), N=number of valid images

        # Normalize for cosine similarity (paper Sec.2.1)
        valid_emb_norm = F.normalize(valid_emb, p=2, dim=-1)
        plane_protos_norm = F.normalize(self.plane_prototypes, p=2, dim=-1)

        # Similarity matrix S (paper Eq.(2))
        similarity_matrix = valid_emb_norm @ plane_protos_norm.t()  # (N, K)

        # Sinkhorn assignment (paper Eq.(2))
        sim_exp = torch.exp(similarity_matrix / self.sinkhorn_eps).t()  # (K, N)
        assignment_code = self.sinkhorn(sim_exp, self.sinkhorn_iter)  # (N, K)

        # Prototype loss (paper Eq.(3))
        logit_scale = self.logit_scale.exp()
        log_prob = F.log_softmax(similarity_matrix * logit_scale, dim=1)
        loss_proto = -(assignment_code * log_prob).sum(dim=1).mean()

        # Reconstruct full assignment matrix (include PAD positions)
        full_assignment_matrix = torch.zeros((B * L, self.K), device=batch_padded_emb.device)
        full_assignment_matrix[flat_valid_mask] = assignment_code
        assignment_matrix = full_assignment_matrix.view(B, L, self.K)

        return loss_proto, assignment_matrix


    def _aggregate_to_plane_representations(self, batch_padded_emb: torch.Tensor, 
                                           batch_pad_mask: torch.Tensor, assignment_matrix: torch.Tensor):
        """Aggregate image embeddings to plane representations (paper Eq.(4)).
        
        Args:
            batch_padded_emb: Padded embeddings (B, L, D)
            batch_pad_mask: Padding mask (B, L), True=PAD
            assignment_matrix: Sinkhorn assignment (B, L, K)
        Returns:
            plane_repr: Aggregated plane representations (B, K, D)
            cluster_valid_mask: Valid plane mask (B, K), True=has valid images
        """
        B, L, D = batch_padded_emb.shape

        # Zero-out PAD positions in assignment matrix
        valid_mask = (~batch_pad_mask).float()  # (B, L)
        assignment_matrix = assignment_matrix * valid_mask.unsqueeze(-1)

        # Weighted sum aggregation (paper Eq.(4))
        numerator = torch.einsum("blk,bld->bkd", assignment_matrix, batch_padded_emb)
        denominator = assignment_matrix.sum(dim=1).unsqueeze(-1) + 1e-6  # δ for numerical stability
        plane_repr = numerator / denominator  # (B, K, D)

        # Identify valid planes (non-negligible assignment sum)
        cluster_valid_mask = (denominator.squeeze(-1) > 0.01)  # (B, K)
        plane_repr = plane_repr * cluster_valid_mask.unsqueeze(-1)

        return plane_repr, cluster_valid_mask


    def _disease_aware_plane_selection(self, plane_repr: torch.Tensor, cluster_valid_mask: torch.Tensor):
        """Disease-aware sparse plane selection (paper Sec.2.2).
        
        Args:
            plane_repr: Aggregated plane representations (B, K, D)
            cluster_valid_mask: Valid plane mask (B, K), True=valid
        Returns:
            disease_embeddings: Aggregated disease embeddings (B, C, D)
            sparse_attention: Sparse attention weights (B, C, K)
        """
        B, K, D = plane_repr.shape
        C = self.C

        # Get disease queries (Q in paper Eq.(5))
        Q = self.disease_queries.squeeze(0)  # (C, D)

        # Normalize for cosine similarity (paper Eq.(6))
        plane_repr_norm = F.normalize(plane_repr, p=2, dim=-1)
        disease_queries_norm = F.normalize(Q, p=2, dim=-1)

        # Relevance scores (paper Eq.(6) + Eq.(7))
        relevance_scores = torch.einsum("cd,bkd->bck", disease_queries_norm, plane_repr_norm)  # (B, C, K)
        relevance_scores = relevance_scores + self.disease_bias_matrix.unsqueeze(0)  # Add bias B

        # Mask invalid planes (set to very low value)
        relevance_scores = relevance_scores.masked_fill(~cluster_valid_mask.unsqueeze(1), -1e9)

        # Top-k plane selection (paper Eq.(8))
        topk_scores, topk_indices = torch.topk(relevance_scores, self.topk_planes, dim=-1)

        # Attention weights (softmax with temperature T)
        attention_weights = torch.softmax(topk_scores / self.attn_T, dim=-1)  # (B, C, topk)

        # Gather selected plane representations
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        plane_repr_expanded = plane_repr.unsqueeze(1).expand(-1, C, -1, -1)
        selected_planes = torch.gather(plane_repr_expanded, dim=2, index=idx_expanded)  # (B, C, topk, D)

        # Aggregate to disease embeddings (paper Eq.(8))
        disease_embeddings = (attention_weights.unsqueeze(-1) * selected_planes).sum(dim=2)  # (B, C, D)

        # Build sparse attention matrix (for visualization)
        sparse_attention = torch.zeros((B, C, K), device=plane_repr.device)
        sparse_attention.scatter_(dim=2, index=topk_indices, src=attention_weights)

        return disease_embeddings, sparse_attention



    # --------------------------
    # Sinkhorn (unchanged)
    # --------------------------
    def sinkhorn(self, Q, nmb_iters):
        """
        Q: (num_prototypes, batch_size) = (K,N)
        returns: (N,K)
        """
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape
            r = torch.ones(K, device=Q.device) / K
            c = torch.ones(B, device=Q.device) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / (u + 1e-6)).unsqueeze(1)
                Q *= (c / (torch.sum(Q, dim=0) + 1e-6)).unsqueeze(0)

            return (Q / (torch.sum(Q, dim=0, keepdim=True) + 1e-6)).t().float()


