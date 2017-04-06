# Tabular based agents
python3 main.py --agent_name TabularQAgent --best
python3 main.py --agent_name BackwardTabularQAgent --best
python3 main.py --agent_name TabularQERAgent --best
python3 main.py --agent_name TabularFixedQERAgent --best
python3 main.py --agent_name TabularQOfflineERAgent --best
python3 main.py --agent_name TabularFixedQOfflineERAgent  --best
python3 main.py --agent_name BackwardTabularFixedQOfflineERAgent --best

# Value approximation based agent
python3 main.py --agent_name DeepTDAgent --best
python3 main.py --agent_name DQNAgent --best
python3 main.py --agent_name DDQNAgent --best
python3 main.py --agent_name DeepFixedQOfflineERAgent --best

# Policy approximation based Agent
python3 main.py --agent_name DeepMCPolicyAgent --best

# Actor critic approximation based agent
python3 main.py --agent_name MCActorCriticAgent --best
python3 main.py --agent_name ActorCriticAgent --best
python3 main.py --agent_name A2CAgent --best
python3 main.py --agent_name TDACAgent --best