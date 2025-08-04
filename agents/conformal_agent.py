from .base_agent import * 
import json 
import numpy as np
import torch 

class ConformalAgent(BaseAgent):
    def __init__(self, model_key: str = "qwen-vl-max", alpha = 0.1, data_path = None):
        """
        Initialize a conformal agent.

        Args:
            model_key: LLM model to use
        """
        super().__init__(AgentType.CONFORMAL_PREDICTOR, model_key)
        self.dataset = self.load_dataset()
        self.alpha = alpha

    def load_dataset(self,data_path):
        """
        Load the similarity matrix from the specified path.

        Returns:
            The loaded similarity matrix.
        """

        with open(data_path) as f:
            dataset = json.load(f)
        all_scores = None
        all_targets = None

        letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

        for key in dataset.keys():
            dataset[key]['scores'] = np.array(dataset[key]['scores'])

            letter_targets = dataset[key]['targets']
            num_targets = np.array([letter_to_num[letter] for letter in letter_targets])
            dataset[key]['targets'] = num_targets
            
            if all_scores is not None:
                all_scores = np.concatenate((all_scores, dataset[key]['scores']), axis=0)
                all_targets = np.concatenate((all_targets, dataset[key]['targets']), axis=0)
            else:
                all_scores = dataset[key]['scores']
                all_targets = dataset[key]['targets']
        
        return dataset
    
    def inference_lac(self, scores, qhat, allow_empty_sets=False):
        """
        Makes prediction sets on new test data
        
        Arguments:
            scores: softmax scores of the test set
            qhat: estimated quantile of the calibration set from the `calirbate_lac` function
            allow_empty_sets: if True allow a prediction set to contain no predictions (will then satisfy upper bound of marginal coverage)

        Returns:
        prediction_sets: boolean mask of prediction sets (True if class is included in the prediction set; otherwise False)
        """
        scores = torch.tensor(scores, dtype=torch.float)
        n = scores.size(0)

        elements_mask = scores > (1 - qhat)

        prediction_sets = elements_mask
        return prediction_sets

    def calibrate_lac(self, scores, targets, return_dist=False):
        """
        Estimates the 1-alpha quantile on held-out calibration data.
        The score function is `1 - max(softmax_score)`.
        
        Arguments:
            scores: softmax scores of the calibration set
            targets: corresponding labels of the calibration set
            alpha: parameter for the desired coverage level (1-alpha)

        Returns:
        qhat: the estimated quantile
        score_dist: the score distribution
        """
        scores = torch.tensor(scores, dtype=torch.float)
        targets = torch.tensor(targets)
        '''
        scores: torch.tensor(Batch_size, 4), softmax scores: 0~1
        targets: torch.tensor(Batch_size), choices:[0,1,2,3]
        '''
        
        assert scores.size(0) == targets.size(0)
        assert targets.size(0)
        n = torch.tensor(targets.size(0))
        assert n

        score_dist = torch.take_along_dim(1 - scores, targets.unsqueeze(1), 1).flatten()
        
        assert (
            0 <= torch.ceil((n + 1) * (1 - self.alpha)) / n <= 1
        )

        score_dist, _ = torch.sort(score_dist, descending=False)

        qhat = torch.quantile(
            score_dist, torch.ceil((n + 1) * (1 - self.alpha)) / n, interpolation="higher"
        )

        return (qhat, score_dist) if return_dist else qhat

    def conformal_prediction(self, val_score, topK=200):

        cal_scores = []
        cal_targets = []
        
        for _, results in self.dataset.items():
            other_score = results['scores']
            other_target = results['targets']
            cal_scores.append(other_score)
            cal_targets.append(other_target)

        cal_scores = np.concatenate(cal_scores, axis=0)
        cal_targets = np.concatenate(cal_targets, axis=0)
        
        index = np.arange(len(cal_scores))
        np.random.shuffle(index)
        index = index[:topK]
        cal_scores = cal_scores[index]
        cal_targets = cal_targets[index]
            
        q = self.calibrate_lac(cal_scores, cal_targets)
        pset = self.inference_lac(val_score, q)
        return pset           
    