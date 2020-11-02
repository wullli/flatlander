from flatland.envs.agent_utils import RailAgentStatus


def is_done(agent):
    return agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED


def get_agent_pos(agent):
    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_virtual_position = agent.position
    elif agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
        agent_virtual_position = agent.target
    else:
        return None
    return agent_virtual_position


def get_save_agent_pos(agent):
    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_virtual_position = agent.position
    else:
        agent_virtual_position = agent.target
    return agent_virtual_position
