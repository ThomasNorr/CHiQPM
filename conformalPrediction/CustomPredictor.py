import torch
from torchcp.classification.predictors import SplitPredictor

from conformalPrediction.cleanScoreFunction import HieraDiffNonConformityScore



class CustomHierarchicalConformityScorePredictor(SplitPredictor):
    def __init__(self, weight):
        # Finetune NoRandom Alph0.925_DDiffAdaX_10_Acc
        score = HieraDiffNonConformityScore(weight)
        super().__init__(score)

    def calculate_threshold(self, logits, labels, alpha, features):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        features = features.to(self._device)
        self.q_hat = 0

        alpha, mask = self.score_function.calibrate_alpha(logits, labels, alpha, features)
        logits = logits[mask]
        labels = labels[mask]
        features = features[mask]
        scores = self.score_function(logits, features, labels, )
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def predict_with_logits(self, logits, features, q_hat=None):
        """
        The input of score function is softmax probability.
        if q_hat is not given by the function 'self.calibrate', the construction progress of prediction set is a naive method.

        :param logits: model output before softmax.
        :param q_hat: the conformal threshold.

        :return: prediction sets
        """

        scores = self.score_function(logits, features)
        scores = scores.to(self._device)
        if q_hat is None:
            q_hat = self.q_hat

        S = self._generate_prediction_set(scores, q_hat)
        return S
    
    def get_level(self, alpha, total_accs):
        """
        Select highest level where accuracy >= (1 - alpha).
        
        Args:
            alpha: significance level (not accuracy!)
            total_accs: tensor of accuracies per level
        
        Returns:
            level index (int)
        """
        sufficient_acc = total_accs >= (1 - alpha)
        return self.score_function._find_highest_sufficient_level(sufficient_acc)
        
    def calibrate_all_levels(self, logits, labels, features):
        """
        Pre-compute all data needed for fast alpha switching.
        Call this ONCE after initialization.
        
        Args:
            logits: calibration logits
            labels: calibration labels
            features: calibration features
        """
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        features = features.to(self._device)
        
        # Populate total_accs (dummy alpha=0.5, we just need the accs)
        self.score_function.calibrate_alpha(logits, labels, 0.5, features)
        
        # Compute scores for all levels
        self.scores_per_level = self.score_function.compute_scores_for_all_levels(
            logits, labels, features
        )

    def get_level_qhat(self, alpha):
        """
        Fast lookup of level and quantile for given alpha.
        Requires calibrate_all_levels() to have been called first.
        
        Args:
            alpha: significance level (1 - accuracy)
            
        Returns:
            (level, q_hat): hierarchical level and conformal quantile
        """
        if not hasattr(self, 'scores_per_level'):
            raise RuntimeError("Must call calibrate_all_levels() before get_level_qhat()")
        
        # Select highest level under which alpha can be reached
        level = self.get_level(alpha, self.score_function.total_accs)
        
        # Get pre-computed scores for that level
        scores = self.scores_per_level[level]
        
        # Compute quantile
        q_hat = self._calculate_conformal_value(scores, alpha)
        
        return level, q_hat
    
    def update_predictor(self, alpha):
        """
        Update predictor state for new alpha without recalibration.
        
        Args:
            alpha: significance level (1 - accuracy)
        """
        level, q_hat = self.get_level_qhat(alpha)
        self.q_hat = q_hat
        self.score_function.level = level