from typing import Dict, List

from ray.rllib.agents.sac.sac_tf_model import SACTFModel
from ray.rllib.utils.framework import TensorType


class SacTtf(SACTFModel):
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (
    TensorType, List[TensorType]):
        pass

    def get_q_values(self, model_out, actions=None):
        """Return the Q estimates for the most recent forward pass.

        This implements Q(s, a).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim]. If None (discrete action
                case), return Q-values for all actions.

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        if actions is not None:
            return self.q_net([model_out, actions])
        else:
            return self.q_net(model_out)

    def get_twin_q_values(self, model_out, actions=None):
        """Same as get_q_values but using the twin Q net.

        This implements the twin Q(s, a).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim]. If None (discrete action
                case), return Q-values for all actions.

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        if actions is not None:
            return self.twin_q_net([model_out, actions])
        else:
            return self.twin_q_net(model_out)

    def get_policy_output(self, model_out):
        """Return the action output for the most recent forward pass.

        This outputs the support for pi(s). For continuous action spaces, this
        is the action directly. For discrete, is is the mean / std dev.

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].

        Returns:
            tensor of shape [BATCH_SIZE, action_out_size]
        """
        return self.action_model(model_out)

    def policy_variables(self):
        """Return the list of variables for the policy net."""

        return list(self.action_model.variables)

    def q_variables(self):
        """Return the list of variables for Q / twin Q nets."""

        return self.q_net.variables + (self.twin_q_net.variables
                                       if self.twin_q_net else [])

