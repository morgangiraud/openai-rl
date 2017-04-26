# Tabular based agents
python3 main.py --best --agent_name TabularQAgent
python3 main.py --best --agent_name BackwardTabularQAgent
python3 main.py --best --agent_name TabularQERAgent
python3 main.py --best --agent_name TabularFixedQERAgent
python3 main.py --best --agent_name TabularQOfflineERAgent
python3 main.py --best --agent_name TabularFixedQOfflineERAgent 
python3 main.py --best --agent_name BackwardTabularFixedQOfflineERAgent

# Value approximation based agent
python3 main.py --best --agent_name DeepTDAgent
python3 main.py --best --agent_name DQNAgent
python3 main.py --best --agent_name DDQNAgent
python3 main.py --best --agent_name DeepFixedQOfflineERAgent

# Policy approximation based Agent
python3 main.py --best --agent_name DeepMCPolicyAgent

# Actor critic approximation based agent
python3 main.py --best --agent_name MCActorCriticAgent
python3 main.py --best --agent_name ActorCriticAgent
python3 main.py --best --agent_name A2CAgent
python3 main.py --best --agent_name TDACAgent