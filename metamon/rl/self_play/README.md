# Self-Play

Utility to auto-manage a local ladder of agents for self-play data collection or bulk eval purposes.

Specify the participating agents with a `.yaml` file. Here is an exmaple:

```yaml
defaults:
  team_set: competitive
  battle_backend: metamon
  # if a list, each agent launch will pick a value from the list at random
  checkpoints: [null]
  temperatures: [1.0]
  num_agents: 1  # number of parallel copies to launch per agent

agents:
  # USERNAME:
  #   model_name: SomeModel
  #   checkpoints: [2]  # override default
  #   num_agents: 3  # will launch USERNAME-1, USERNAME-2, USERNAME-3
  
  PAC-MM-Kadabra:
    model_name: Kadabra
  
  PAC-MM-SynRLV2:
    model_name: SyntheticRLV2 
```

```bash
python launch_models.py --format gen2ou --gpus 0 1 --config earlygen_config.yaml
```