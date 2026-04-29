import torch
from torchcp.classification.scores.base import BaseScore


class HieraDiffNonConformityScore(BaseScore):

    def __init__(self, weight):
        self.weight = weight
        self.weights_per_class = int(weight.sum(dim=1).mean().item())
        super().__init__()

    def _find_highest_sufficient_level(self, sufficient_acc):
        if sufficient_acc.any():
            if sufficient_acc.all():
                return len(sufficient_acc)
            else:
                return torch.argmin(sufficient_acc.float()).item()
        else:
            print("Warning: No level satisfies the accuracy constraint")
            return 0

    def compute_scores_for_all_levels(self, logits, labels, features):
        """
        Pre-compute nonconformity scores for all hierarchical levels.
        
        Args:
            logits: calibration logits
            labels: calibration labels  
            features: calibration features
            
        Returns:
            dict: {level: scores_tensor}
        """
        original_level = getattr(self, 'level', 0)
        
        scores_per_level = {}
        for level in range(self.weights_per_class + 1):
            self.level = level
            scores = self(logits, features, labels)
            scores_per_level[level] = scores
        
        self.level = original_level
        
        return scores_per_level

    def get_adaptive_level(self, logits, labels, alpha, features):
        """
        Use calibration data to get fix level accuracy values.
        accuracy levels = [# of correct predictions with level l / # of samples for l in levels]
        """
        delta_n,_= self.get_delta_n_up(logits, features)
        correct_label_included = delta_n[torch.arange(len(labels)), :,labels]
        total_corrects = correct_label_included.sum(dim=0)
        self.total_accs = total_corrects / len(labels)
        sufficient_acc = self.total_accs >= (1 - alpha)

        shared_level = self._find_highest_sufficient_level(sufficient_acc)
        return shared_level

    def calibrate_alpha(self, logits, labels, alpha, features):

        self.level = self.get_adaptive_level(logits, labels, alpha, features)
        mask = torch.ones(logits.shape[0], device=logits.device, dtype=torch.bool)

        return alpha, mask

    def _calculate_all_label(self, logits, features):
        return self.nonconformity_score_for_every_class(logits, features)

    def get_sorted_indices_and_values(self, features):
        # eq. 7
        masked_features = features[..., None] * self.weight.T + torch.ones_like(features[..., None]) * -1 * (
                1 - self.weight.T)
        sorted_values, sorted_indices = torch.sort(masked_features, dim=1, descending=True)
        sorted_values, sorted_indices = sorted_values[:, :self.weights_per_class], sorted_indices[
            :, :self.weights_per_class]
        return sorted_values, sorted_indices

    def get_shared_with_preds_from_sorted(self,  sorted_indices, actual_preds):
        # eq. 8
        if isinstance(actual_preds,int): # during Explanation generation
            label_indices = sorted_indices[:, :, actual_preds]

        else:

            label_indices = sorted_indices[torch.arange(len(actual_preds)), :, actual_preds]

        same_as_pred_feat = label_indices[..., None] == sorted_indices
        delta_n = torch.cumprod(same_as_pred_feat, dim=1).bool()
        return delta_n

    def get_delta_n_up(self, logits, features):
        sorted_values, sorted_indices = self.get_sorted_indices_and_values(features)
        actual_preds = torch.argmax(logits, dim=1)
        delta_n = self.get_shared_with_preds_from_sorted( sorted_indices, actual_preds)
        return delta_n,sorted_values

    def nonconformity_score_for_every_class(self, logits, features):
        logits = logits.to(self.weight.device)
        features = features.to(self.weight.device)

        delta_n, sorted_values = self.get_delta_n_up(logits, features)   # [B, K, C]
        B, K, C = sorted_values.shape
        device = sorted_values.device

        # Eq. (3) short paper / Eq. (10) long paper: shared-path terms delta_j^c * f^*_{F_j^c}.
        masked_for_upscore = sorted_values * delta_n.float()

        # Eq. (4) short / Eq. (11) long: first mismatch position i_div = F^c_{k+1}.
        mismatch_position = torch.argmin(delta_n.float(), dim=1)         # [B, C]
        contains_mismatch = (delta_n.sum(dim=1) < self.weights_per_class)

        batch_idx = torch.arange(B, device=device).view(B, 1)
        class_idx = torch.arange(C, device=device).view(1, C)

        # Diversion value f^*_{i_div}; the minus sign is applied when building s_sel.
        value_at_diversion = (
            sorted_values[batch_idx, mismatch_position, class_idx] * contains_mismatch
        )

        level = int(self.level)
        if level < 0 or level > self.weights_per_class:
            raise ValueError(f"level must be in [0, {self.weights_per_class}], got {level}")

        # Eq. (6) short / Eq. (13) long: indicator delta^c_{n_limit}.
        # Paper sum upper bound is n_wc - 1 (inclusive, 1-based).
        # In 0-based Python slicing, this corresponds to exclusive end index n_wc.
        if level == 0:
            limit_delta = torch.ones(B, C, device=device)
            # sum_{j=1}^{...} for global case
            limited_shared_sum = masked_for_upscore.sum(dim=1)
        else:
            # Level n means n shared features required => index n-1 in delta_n.
            limit_delta = delta_n[:, level - 1].float()
            # sum from j=1+n_limit (paper) => 0-based slice starts at index level
            limited_shared_sum = masked_for_upscore[:, level:].sum(dim=1)

        # Eq. (5) short / Eq. (12) long inside the limited-score parentheses.
        limited_s_sel = -value_at_diversion - limited_shared_sum

        # Eq. (6) short / Eq. (13) long final score.
        s = limit_delta * limited_s_sel
        return s

    def _calculate_single_label(self, logits, features, labels):
        limited_up_score_masked = self.nonconformity_score_for_every_class(logits, features)
        nonconformity_scores_of_target = limited_up_score_masked[torch.arange(len(labels)), labels]
        return nonconformity_scores_of_target


    def __call__(self, logits, features, label=None):
        assert len(logits.shape) <= 2, "dimension of logits are at most 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if label is None:
            return self._calculate_all_label(probs, features)
        else:
            return self._calculate_single_label(probs, features, label)