from agents.basic_agent import BasicAgent
from agents.tabular_q_agent import TabularQAgent, TabularQERAgent, TabularFixedQERAgent, TabularQplusAgent, TabularFixedQplusAgent, BackwardTabularQAgent, BackwardTabularFixedQplusAgent
from agents.deep_q_agent import DeepFixedQPlusAgent, DQNAgent
from agents.deep_policy_agent import DeepMCPolicyAgent, ActorCriticAgent

__all__ = [
    "BasicAgent",
    "TabularQAgent",
    "TabularQERAgent",
    "TabularFixedQERAgent",
    "TabularQplusAgent",
    "TabularFixedQplusAgent",
    "BackwardTabularQAgent",
    "BackwardTabularFixedQplusAgent",
    "DeepFixedQPlusAgent",
    "DQNAgent",
    "DeepMCPolicyAgent",
    "ActorCriticAgent",
]

def make_agent(config, env):
    if(config['agent_name'] == 'TabularQAgent'):
        return TabularQAgent(config, env)
    elif(config['agent_name'] == 'BackwardTabularQAgent'):
        return BackwardTabularQAgent(config, env)
    elif(config['agent_name'] == 'TabularQERAgent'):
        return TabularQERAgent(config, env)
    elif(config['agent_name'] == 'TabularFixedQERAgent'):
        return TabularFixedQERAgent(config, env)
    elif(config['agent_name'] == 'TabularQplusAgent'):
        return TabularQplusAgent(config, env)
    elif(config['agent_name'] == 'TabularFixedQplusAgent'):
        return TabularFixedQplusAgent(config, env)
    elif(config['agent_name'] == 'BackwardTabularFixedQplusAgent'):
        return BackwardTabularFixedQplusAgent(config, env)
    elif(config['agent_name'] == 'DeepFixedQPlusAgent'):
        return DeepFixedQPlusAgent(config, env)
    elif(config['agent_name'] == 'DQNAgent'):
        return DQNAgent(config, env)
    elif(config['agent_name'] == 'DeepMCPolicyAgent'):
        return DeepMCPolicyAgent(config, env)
    elif(config['agent_name'] == 'ActorCriticAgent'):
        return ActorCriticAgent(config, env)
    else:
        raise Exception('The agent name %s does not exist' % config['agent_name'])