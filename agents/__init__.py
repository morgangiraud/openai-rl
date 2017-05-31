from agents.basic_agent import BasicAgent
from agents.tabular_agent import TabularMCAgent, TabularTD0Agent, TabularNStepAgent, TabularLambdaAgent, TabularQAgent, BackwardTabularQAgent, TabularQERAgent, TabularFixedQERAgent, TabularQOfflineERAgent, TabularFixedQOfflineERAgent, BackwardTabularFixedQOfflineERAgent
from agents.deep_q_agent import DeepTDAgent, DeepFixedQOfflineERAgent, DQNAgent, DDQNAgent
from agents.deep_policy_agent import DeepMCPolicyAgent, MCActorCriticAgent, ActorCriticAgent, A2CAgent, TDACAgent

__all__ = [
    "BasicAgent",
    "TabularMCAgent",
    "TabularTD0Agent",
    "TabularNStepAgent",
    "TabularLambdaAgent",
    "TabularQAgent",
    "BackwardTabularQAgent",
    "TabularQERAgent",
    "TabularFixedQERAgent",
    "TabularQOfflineERAgent",
    "TabularFixedQOfflineERAgent",
    "BackwardTabularFixedQOfflineERAgent",
    "DeepTDAgent",
    "DeepFixedQOfflineERAgent",
    "DQNAgent",
    "DDQNAgent",
    "DeepMCPolicyAgent",
    "MCActorCriticAgent",
    "ActorCriticAgent",
    "A2CAgent",
    "TDACAgent",
]

def make_agent(config, env):
    if config['agent_name'] in __all__:
        return globals()[config['agent_name']](config, env)
    else:
        raise Exception('The agent name %s does not exist' % config['agent_name'])

def get_agent_class(config):
    if config['agent_name'] in __all__:
        return globals()[config['agent_name']]
    else:
        raise Exception('The agent name %s does not exist' % config['agent_name'])