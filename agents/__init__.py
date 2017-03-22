from agents.basic_agent import BasicAgent
from agents.tabular_q_agent import TabularQAgent, TabularQplusAgent, TabularFixedQplusAgent, BackwardTabularQAgent, BackwardTabularFixedQplusAgent
from agents.deep_q_agent import DeepQAgent

__all__ = [
    "BasicAgent",
    "TabularQAgent",
    "TabularQplusAgent",
    "TabularFixedQplusAgent",
    "BackwardTabularQAgent",
    "BackwardTabularFixedQplusAgent",
    "DeepQAgent"
]

def make_agent(config, env):
    if(config['agent_name'] == 'TabularQAgent'):
        return TabularQAgent(config, env)
    elif(config['agent_name'] == 'BackwardTabularQAgent'):
        return BackwardTabularQAgent(config, env)
    elif(config['agent_name'] == 'TabularQplusAgent'):
        return TabularQplusAgent(config, env)
    elif(config['agent_name'] == 'TabularFixedQplusAgent'):
        return TabularFixedQplusAgent(config, env)
    elif(config['agent_name'] == 'BackwardTabularFixedQplusAgent'):
        return BackwardTabularFixedQplusAgent(config, env)
    elif(config['agent_name'] == 'DeepQAgent'):
        return DeepQAgent(config, env)
    else:
        raise Exception('The agent name %s does not exist' % config['agent_name'])