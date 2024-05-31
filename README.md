# Multi-echelon Supply Chain Optimization

Este projeto implementa um agente de aprendizado por reforço profundo (PPO) para otimizar operações em uma cadeia de suprimentos multi-echelon com demandas sazonais incertas e tempos de entrega variáveis.

## Estrutura do Projeto

- `data/`: Scripts e dados para geração e processamento de datasets.
- `models/`: Definições dos agentes (PPO e LP) e simulação do ambiente.
- `experiments/`: Scripts para treinamento e avaliação dos agentes.
- `results/`: Logs, modelos treinados, gráficos e resumos dos resultados.
- `utils/`: Funções utilitárias para manipulação de dados, modelos e geração de gráficos.
- `main.py`: Script principal para executar o treinamento e a avaliação.
- `requirements.txt`: Lista de dependências do projeto.
- `README.md`: Documentação do projeto.

## Configuração do Ambiente Virtual

Para configurar e ativar o ambiente virtual no Windows, siga os passos abaixo:

### Passo 1: Criar e ativar o ambiente virtual

Navegue até o diretório do projeto e execute o comando abaixo para criar o ambiente virtual com Anaconda:

```bash
conda create -n venvsupplyrl python=3.7
```

Em seguida, execute o comando abaixo para ativar o ambiente virtual:

```bash
conda activate venvsupplyrl
```
Por fim execute o comando abaixo para instalar as dependencias:

```bash
pip install -r requirements.txt
```

## Como Executar

1. Gere os datasets:
    ```bash
    python data/make_dataset.py
    ```

2. Treine o agente PPO:
    ```bash
    python main.py
    ```

3. Avalei o modelo treinado:
    ```bash
    python experiments/evaluate.py
    ```


## Configurações

As configurações dos experimentos podem ser ajustadas no arquivo `experiments/config.yaml`.

## Estrutura dos Dados

Os dados de demanda são gerados usando uma função senoidal perturbada para simular incertezas sazonais.

## Metodologia

A metodologia segue a abordagem descrita no artigo "Multi-echelon Supply Chains with Uncertain Seasonal Demands and Lead Times Using Deep Reinforcement Learning".

