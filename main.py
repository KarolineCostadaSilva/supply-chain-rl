import torch
import pandas as pd
from experiments.train_ppo import train_ppo as train_ppo
from experiments.evaluate import evaluate
from models.lp_agent import LPAgent
from models.environment import SupplyChainEnv
from utils.plotting import plot_scenario

def main():
    scenarios = ['N0', 'N20', 'N40', 'N60', 'N0cl', 'N20cl', 'N40cl', 'N60cl', 'rN0', 'rN50', 'rN100', 'rN200', 'rN0cl', 'rN50cl', 'rN100cl', 'rN200cl', 'N20stc']

    # # Verificando se há GPU
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # else:
    #     device = torch.device('cpu')
    #     print("No GPU available, using CPU")
    
    # Treinamento e avaliação do PPO após resolver o LP
    # print(f"Training PPO for scenario {scenario}")
    # train_ppo('experiments/config.yaml', scenario)
    
    # print(f"Evaluating PPO for scenario {scenario}")
    # evaluate('experiments/config.yaml', scenario)

    # Plotando gráficos dos dados
    # plot_scenario('N20')
    # plot_scenario('N60')

    results = []
    for scenario in scenarios:
        print(f"Evaluating for scenario {scenario}")
        result = evaluate(scenario)  # Chama a função evaluate para cada cenário
        results.append(result)
        print(f"Results for {scenario}: Cost = {result['lp_cost']}")

    # Salvando resultados em um arquivo CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/scenario_evaluation_lp.csv', index=False)
    print("Evaluation results saved to 'scenario_evaluation_lp.csv'.")

if __name__ == "__main__":
    main()