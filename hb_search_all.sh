agentList="
  TabularQAgent
  BackwardTabularQAgent
  TabularQERAgent
  TabularFixedQERAgent
  TabularQOfflineERAgent
  TabularFixedQOfflineERAgent
  BackwardTabularFixedQOfflineERAgent

  DeepTDAgent
  DQNAgent
  DDQNAgent
  DeepFixedQOfflineERAgent

  DeepMCPolicyAgent
  MCActorCriticAgent
  ActorCriticAgent
  A2CAgent
  TDACAgent
"

for agent in $agentList; do
  python3 main.py --full_search --agent_name "$agent" --hb_nb_process 3
done
