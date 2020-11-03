from flatland.envs.agent_utils import RailAgentStatus, EnvAgent


def get_virtual_position(agent: EnvAgent):
    agent_virtual_position = agent.position
    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    return agent_virtual_position
