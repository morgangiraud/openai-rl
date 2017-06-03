agentList="
  TabularMCAgent
  TabularTD0Agent
  TabularTDNAgent
  TabularTDLambdaAgent
  TabularQAgent
  TabularQLambdaBackwardAgent
  TabularQERAgent
  TabularQDoubleERAgent
  TabularExpectedSarsaAgent
  TabularSigmaAgent
  TabularSigmaLambdaBackwardAgent

  DeepTDAgent
  DQNAgent
  DDQNAgent
  
  DeepMCPolicyAgent
  MCActorCriticAgent
  ActorCriticAgent
  A2CAgent
  TDACAgent
"

for agent in $agentList; do
  python3 main.py --fullsearch --agent_name "$agent" --nb_process 3
done
