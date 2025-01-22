import torch
import torch.nn as nn

class FeaturesLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(FeaturesLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        """
        Вычисляет contrastive loss.
        
        :param features: Tensor, размер (batch_size, feature_dim), векторы признаков объектов.
        :param labels: Tensor, размер (batch_size), реальные метки объектов.
        :return: Tensor, значение функции потерь.
        """
        batch_size = features.size(0)
        loss = 0.0
        positive_pairs = 0
        negative_pairs = 0

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:  
                    distance = torch.norm(features[i] - features[j], p=2) ** 2
                    if labels[i] == labels[j]:  
                        loss += distance
                        positive_pairs += 1
                    else:  
                        loss += torch.clamp(self.margin - torch.sqrt(distance), min=0) ** 2
                        negative_pairs += 1
        
        if positive_pairs > 0:
            loss /= positive_pairs + negative_pairs

        return loss
