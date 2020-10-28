from ray.rllib.evaluation import MultiAgentEpisode


def on_episode_end(info):
    episode: MultiAgentEpisode = info["episode"]

    episode_steps = 0
    episode_max_steps = 0
    episode_num_agents = 0
    episode_score = 0
    episode_done_agents = 0
    episode_num_swaps = 0

    try:
        for agent, agent_info in episode._agent_to_last_info.items():
            if 'agent' in str(agent) or isinstance(agent, int):
                if episode_max_steps == 0:
                    episode_max_steps = agent_info["max_episode_steps"]
                    episode_num_agents = agent_info["num_agents"]
                episode_steps = max(episode_steps, agent_info["agent_step"])
                episode_score += agent_info["agent_score"]
                if "num_swaps" in agent_info:
                    episode_num_swaps += agent_info["num_swaps"]
                if agent_info["agent_done"]:
                    episode_done_agents += 1
    except:
        for agent, agent_info in enumerate(episode._agent_to_last_info["group_1"]["_group_info"]):
            if episode_max_steps == 0:
                episode_max_steps = agent_info["max_episode_steps"]
                episode_num_agents = agent_info["num_agents"]
            episode_steps = max(episode_steps, agent_info["agent_step"])
            episode_score += agent_info["agent_score"]
            if "num_swaps" in agent_info:
                episode_num_swaps += agent_info["num_swaps"]
            if agent_info["agent_done"]:
                episode_done_agents += 1

    norm_factor = 1.0 / (episode_max_steps * episode_num_agents)
    percentage_complete = float(episode_done_agents) / episode_num_agents

    episode.custom_metrics["episode_steps"] = episode_steps
    episode.custom_metrics["episode_max_steps"] = episode_max_steps
    episode.custom_metrics["episode_num_agents"] = episode_num_agents
    episode.custom_metrics["episode_return"] = episode.total_reward
    episode.custom_metrics["episode_score"] = episode_score
    episode.custom_metrics["episode_score_normalized"] = episode_score * norm_factor
    episode.custom_metrics["episode_num_swaps"] = episode_num_swaps / 2
    episode.custom_metrics["percentage_complete"] = percentage_complete
