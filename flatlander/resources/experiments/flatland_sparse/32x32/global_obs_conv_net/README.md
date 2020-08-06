# Global observation convnet experiments

https://app.wandb.ai/masterscrat/flatland/reports/Flatland-Sparse-32x32--Vmlldzo5MzQ3Nw/edit

## Method

In this experiment, we compare the performance of two established CNN architectures on the global 
observations. In the first case, agents are based on the Nature-CNN architecture [2] that 
consists of 3 convolutional layers followed by a dense layer. In the second case, the 
agents are based on the IMPALA-CNN [1] network, which consists of a 15-layer residual architecture 
neural network followed by a dense layer. Agents share the same centralized
policy network.

## Results

TODO

## Plots

TODO

## Conclusion

TODO

## Refrences

[1] Lasse Espeholt et al. “IMPALA: Scalable Distributed Deep-RL with Importance
Weighted Actor-Learner Architectures”. In: Proceedings of the 35th International
Conference on Machine Learning. Vol. 80. 2018, pp. 1407–1416. URL: [https://arxiv.org/abs/1802.01561](https://arxiv.org/abs/1802.01561)

[2] Volodymyr Mnih et al. “Human-level control through deep reinforcement learn-
ing”. In: Nature 518.7540 (2015), pp. 529–533. issn: 1476-4687. doi: 10 . 1038 /
nature14236. URL: [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)


