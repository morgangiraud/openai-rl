from agents.basic_agent import BasicAgent, TabularBasicAgent
from agents.tabular_mc_agent import TabularMCAgent
from agents.tabular_td_0_agent import TabularTD0Agent
from agents.tabular_td_0_nstep_agent import TabularNStepTD0Agent
from agents.tabular_td_lambda_agent import TabularTDLambdaAgent
from agents.tabular_q_agent import TabularQAgent
from agents.tabular_q_lambda_backward_agent import TabularQLambdaBackwardAgent
from agents.tabular_q_er_agent import TabularQERAgent
from agents.tabular_q_double_er_agent import TabularQDoubleERAgent
from agents.tabular_expected_sarsa_agent import TabularExpectedSarsaAgent
from agents.tabular_sigma_agent import TabularSigmaAgent
from agents.tabular_sigma_lambda_backward_agent import TabularSigmaLambdaBackwardAgent

from agents.deep_q_agent import DeepTDAgent, DQNAgent, DQNTimeAwareAgent, DDQNAgent

from agents.deep_policy_agent import PolicyAgent, ActorQCriticAgent, ActorCriticAgent, A2CAgent, TDACAgent

from agents.cm_agent import CMAgent

__all__ = [
    "BasicAgent",
    "TabularMCAgent",
    "TabularTD0Agent",
    "TabularNStepTD0Agent",
    "TabularTDLambdaAgent",
    "TabularQAgent",
    "TabularQLambdaBackwardAgent",
    "TabularQERAgent",
    "TabularQDoubleERAgent",
    "TabularExpectedSarsaAgent",
    "TabularSigmaAgent",
    "TabularSigmaLambdaBackwardAgent",

    "DeepTDAgent",
    "DQNAgent",
    "DQNTimeAwareAgent",
    "DDQNAgent",

    "PolicyAgent",
    "ActorQCriticAgent",
    "ActorCriticAgent",
    "A2CAgent",
    "TDACAgent",

    "CMAgent",
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