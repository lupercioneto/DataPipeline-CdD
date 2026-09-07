# DataPipeline-CdD

Pipeline reprodutível de dados aplicado ao tema **Saúde Pública (COVID-19)**, construído para
a Avaliação Prática Unificada da disciplina de Ciência de Dados:

- **Parte 1** — infraestrutura de ETL: ingestão, limpeza, imputação de nulos, remoção de
  outliers via IQR e consolidação do dataset (`dados_limpos_final.csv`).
- **Parte 2** — modelagem estatística, testagem empírica e raciocínio causal: Bootstrap e
  Intervalos de Confiança, Teste de Hipóteses (A/B) via Teste de Permutação, Regressão Linear
  Múltipla, Classificação (Regressão Logística vs. KNN) com GridSearchCV, PCA + K-Means, e
  discussão de causalidade.

Dataset utilizado: [Country Vaccinations](https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress) (Kaggle).

---

## 1. Estrutura do Projeto

```
DataPipeline-CdD/
├── data/
│   ├── original/              # CSV bruto baixado do Kaggle (não versionado)
│   └── processed/              # saídas intermediárias
├── src/
│   ├── extract/
│   │   └── extractor.py        # ingestão e leitura dos arquivos
│   ├── transform/
│   │   └── transformer.py      # tratamento estatístico (nulos e IQR)
│   ├── inference/
│   │   ├── bootstrap.py        # reamostragem bootstrap e ICs (Seção 4.1)
│   │   └── ab_testing.py       # teste de permutação e p-valor (Seção 4.2)
│   ├── models/
│   │   ├── regression.py       # regressão linear múltipla (Seção 4.3)
│   │   ├── machine_learning.py # KNN, Reg. Logística e GridSearchCV (Seção 4.3)
│   │   └── unsupervised.py     # PCA e K-Means (Seção 4.4)
│   ├── visualize/
│   │   └── visualize.py        # gráfico de integridade visual
│   ├── logs/
│   │   └── log.py              # logger simples
│   └── main.py                 # orquestrador de todo o pipeline (ETL + Parte 2)
├── plots/
│   ├── dados_vacinacao_diaria.png  # evolução temporal da vacinação diária (Parte 1)
│   ├── distribuicao_bootstrap.png  # histograma bootstrap + limites dos ICs
│   ├── distribuicao_permutacao.png # distribuição sob H0 do teste de permutação
│   ├── curva_cotovelo_kmeans.png   # método do cotovelo (inércia x k)
│   ├── clusters_kmeans.png         # projeção PCA colorida por cluster
│   └── pca_projecao.png            # projeção PCA (PC1 x PC2)
├── dados_limpos_final.csv      # saída final consolidada (Parte 1)
├── requirements.txt
└── README.md
```

Para executar todo o pipeline (ETL + Parte 2): `cd src && python main.py`

Cada módulo de inferência/modelagem também pode ser executado isoladamente (lendo
`dados_limpos_final.csv` já gerado), por exemplo: `cd src && python inference/bootstrap.py`.

---

## 2. Camada de Ingestão — Amostragem e Viés

### 2.1 População-alvo ideal

A população-alvo ideal para este tema seria **todos os indivíduos do planeta elegíveis a receber
a vacina contra a COVID-19, registrados de forma diária, granular e padronizada, em todos os
países e territórios do mundo**, com discriminação por dose (1ª, 2ª, reforço), faixa etária, e
fabricante do imunizante.

### 2.2 Estrutura de Acesso (Access Frame) real

O dataset disponível no Kaggle é, na prática, uma **agregação diária por país**, coletada por
fontes governamentais e organismos de saúde (ex.: *Our World in Data*) e depois compilada pelo
mantenedor do dataset. Isso significa que o *access frame* real é bem mais restrito que a
população-alvo:

- Os dados estão no nível **país/data**, não no nível indivíduo — perdendo toda a heterogeneidade
  interna (regiões, idade, classe social) de cada país.
- A frequência e a qualidade do reporte dependem da **capacidade administrativa** de cada governo:
  países com sistemas de saúde mais informatizados tendem a reportar com mais regularidade e
  precisão do que países com infraestrutura mais limitada.
- O dataset é uma **compilação de segunda mão** (não é coleta primária), logo herda qualquer
  inconsistência, atraso ou erro de digitação das fontes originais.

### 2.3 Riscos de Viés de Seleção

Existem riscos relevantes de viés de seleção:

- **Viés de cobertura geográfica**: países que não reportam (ou reportam de forma irregular)
  ficam sub-representados ou ausentes, fazendo com que o dataset reflita desproporcionalmente
  países com melhor infraestrutura de dados — não necessariamente os países com mais vacinação
  real.
- **Viés temporal de início de reporte**: cada país começou a registrar dados em datas distintas
  (conforme o início de sua campanha de vacinação), o que distorce comparações diretas de
  "evolução global" se não for tratado com cuidado.
- **Viés de fonte**: como os números vêm de fontes oficiais governamentais, há possibilidade de
  sub-registro estratégico ou atraso de publicação em contextos politicamente sensíveis.

---

## 3. EDA e Tratamento Estatístico

### 3.1 Dicionário de Dados

| Coluna | Classificação Estatística | Descrição |
|---|---|---|
| `country` | Categórica (nominal) | Nome do país |
| `iso_code` | Categórica (nominal) | Código ISO do país |
| `date` | — (temporal) | Data do registro (YYYY-MM-DD) |
| `total_vaccinations` | Discreta | Total acumulado de doses aplicadas |
| `people_vaccinated` | Discreta | Pessoas com ao menos 1 dose |
| `people_fully_vaccinated` | Discreta | Pessoas com esquema vacinal completo |
| `daily_vaccinations` | Discreta | Doses aplicadas no dia (métrica principal) |
| `total_vaccinations_per_hundred` | Contínua | Doses a cada 100 habitantes |
| `people_vaccinated_per_hundred` | Contínua | Pessoas vacinadas a cada 100 habitantes |
| `vaccines` | Categórica (nominal) | Fabricantes utilizados no país |
| `source_name` | Categórica (nominal) | Órgão/fonte de origem do dado |
| `source_website` | Categórica (nominal) | URL da fonte |

> Variáveis contagem de doses (`total_vaccinations`, `daily_vaccinations`, etc.) são tecnicamente
> **discretas** (não fracionárias), enquanto as taxas "per hundred" são **contínuas**, pois
> resultam de uma divisão e podem assumir qualquer valor real dentro de um intervalo.

### 3.2 Tratamento de Nulos — Viés vs. Variância

A estratégia adotada no `transformer.py` (método `handle_nulls`) foi **mista**:

- **Exclusão de registros (`dropna`)** para as colunas `country` e `date`: são colunas-chave de
  identificação; um registro sem elas não tem como ser localizado nem analisado temporalmente,
  então removê-lo é a opção mais segura.
- **Imputação por zero (`fillna(0)`)** para colunas numéricas (ex.: `daily_vaccinations`): a
  ausência de valor, neste contexto, frequentemente indica que **não houve registro de aplicação
  de doses naquele dia** (e não que o dado foi perdido aleatoriamente), então tratar como `0` é
  mais coerente com a semântica do dado do que usar média/mediana.

**Impacto teórico:**

- A exclusão de registros com `country`/`date` ausente **reduz a variância amostral** ao eliminar
  ruído não interpretável, mas também **reduz o tamanho da amostra**, o que pode introduzir viés
  caso a ausência desses campos não seja aleatória (ex.: se um país específico tiver
  sistematicamente mais falhas de reporte, ele fica sub-representado).
- A imputação por `0` em vez de média/mediana **evita inflar artificialmente os valores de
  vacinação diária** (o que aconteceria se usássemos a média), mas pode **subestimar a variância
  real do fenômeno** em dias onde o dado simplesmente não foi coletado a tempo (atraso de reporte
  ≠ ausência de vacinação). Isso introduz um viés sistemático de subestimação em períodos de
  defasagem no reporte.

### 3.3 Outliers — IQR

A função `remove_outliers_iqr` calcula Q1, Q3 e o Intervalo Interquartil (IQR = Q3 − Q1),
removendo registros fora do intervalo `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`. No pipeline, essa rotina é
aplicada à coluna `daily_vaccinations`, isolando dias com picos atípicos de aplicação (que
costumam ocorrer por consolidação retroativa de múltiplos dias em um único registro, distorcendo
a métrica diária real).

---

## 4. Análise de Domínio — Inferência Causal e Ceteris Paribus

**Relação hipotética escolhida:** *Renda per capita do país* (proxy de desenvolvimento
socioeconômico) **e** *Velocidade de vacinação da população* (medida pela evolução de
`people_vaccinated_per_hundred` ao longo do tempo).

### a) Por que correlação não comprova causalidade

Mesmo que se observe estatisticamente que países com maior renda per capita apresentam curvas de
vacinação mais rápidas, essa correlação não prova, por si só, que a renda **causa** maior
velocidade de vacinação. Correlação capta apenas covariação estatística entre duas variáveis,
mas não informa a direção do efeito, nem descarta que ambas estejam sendo influenciadas por uma
terceira variável. Pode haver causalidade reversa (governos investem mais em saúde justamente
por terem economia mais aquecida, criando um ciclo) ou a relação pode ser inteiramente espúria,
mediada por fatores não observados na análise.

### b) Variáveis de confusão (confounders)

1. **Capacidade logística/industrial do país de produzir ou importar vacinas**: países que são
   sede de fabricantes farmacêuticos (ou têm acordos comerciais privilegiados) podem vacinar mais
   rápido independentemente da renda per capita média da população — essa variável influencia
   tanto a "renda" (indiretamente, por ser polo industrial) quanto a "velocidade de vacinação"
   (acesso direto ao insumo).
2. **Qualidade institucional / eficiência do sistema público de saúde**: países com sistemas de
   saúde pública bem estruturados (capacidade de logística de distribuição, rede de postos de
   vacinação, campanhas de conscientização) podem vacinar rapidamente mesmo com renda per capita
   moderada. Essa variável correlaciona-se com renda (países mais ricos tendem a ter sistemas mais
   robustos) mas tem efeito próprio sobre a velocidade de vacinação, confundindo a relação direta.

Outros confounders plausíveis incluem densidade populacional e estrutura etária da população.

### c) Desenho ideal sob Ceteris Paribus

Para isolar o efeito causal puro da renda per capita sobre a velocidade de vacinação, seria
necessário um desenho onde todas as demais variáveis relevantes permanecessem constantes
("tudo o mais constante") entre os grupos comparados. Conceitualmente, isso significaria comparar
países (ou, idealmente, regiões dentro de países) que:

- tivessem **acesso idêntico** ao mesmo lote/fabricante de vacina, no mesmo momento;
- possuíssem **capacidade logística e institucional de saúde equivalente**;
- tivessem **estrutura etária e densidade populacional semelhantes**;

e que diferissem **apenas** na renda per capita. Na prática, esse cenário ideal é dificilmente
alcançável com dados observacionais — exigiria um desenho quase-experimental (ex.: método de
pareamento estatístico — *matching* — entre países/regiões similares em todas as covariáveis
exceto a renda) para aproximar, ainda que de forma imperfeita, o isolamento do efeito causal.

---

## 5. Visualização Científica

O script `visualize.py` gera um **gráfico de linhas** mostrando a evolução temporal da soma global
de `daily_vaccinations` (métrica principal do tema), com:

- eixos identificados (`Data` no eixo X, `Vacinações aplicadas por dia` no eixo Y);
- unidade explícita (doses/dia, em valores absolutos sem notação científica);
- eixo Y iniciando em zero, evitando distorção visual de magnitude;
- sem truncamentos ou escalas não-lineares que possam induzir leitura equivocada.

A imagem é exportada automaticamente para `plots/dados_vacinacao_diaria.png`, a partir da base
já tratada (`dados_limpos_final.csv`).

---

## 6. Saída do Pipeline (Parte 1)

Ao final da execução de `main.py`, o pipeline gera:

- `dados_limpos_final.csv` — base de dados final, tratada e consolidada, na raiz do projeto;
- `plots/dados_vacinacao_diaria.png` — gráfico de integridade visual.

---

## 7. Estimação de Parâmetros e Bootstrap (`bootstrap.py`)

Variável numérica contínua escolhida: **`people_vaccinated_per_hundred`** (pessoas vacinadas a
cada 100 habitantes) — é a taxa de cobertura vacinal mais direta do dataset e assume qualquer
valor real dentro de um intervalo, o que a torna adequada para o exercício.

Com base na amostra (N = 73.316, após limpeza e remoção de outliers) e 2.000 réplicas bootstrap
com reposição, foram obtidos:

| Estatística | Valor |
|---|---|
| Média Amostral (X̄) | 16,5430 |
| Desvio Padrão Amostral (s) | 27,3092 |
| IC 95% Bootstrap (percentil 2,5%/97,5%) | [16,3392 ; 16,7521] |
| IC 95% Paramétrico (X̄ ± 1,96·s/√N) | [16,3453 ; 16,7406] |

Gráfico: [`plots/distribuicao_bootstrap.png`](plots/distribuicao_bootstrap.png).

**Comparação das metodologias:** os dois intervalos são praticamente coincidentes (diferença de
amplitude inferior a 0,02 no limite superior/inferior). Isso é esperado: com N = 73.316, o Erro
Padrão (s/√N ≈ 0,1008) é extremamente pequeno, então tanto a aproximação normal quanto a
distribuição empírica bootstrap convergem para o mesmo intervalo estreito em torno de X̄.

**O TCL se aplica?** Sim. A variável `people_vaccinated_per_hundred` é fortemente assimétrica à
direita (mais da metade das observações são zero — países/dias sem cobertura vacinal iniciada —,
com uma cauda longa até ~125), então a distribuição da variável individual está longe de ser
normal. Entretanto, o Teorema Central do Limite trata da distribuição da **média amostral**, não
da variável original: com N tão grande (>> 30), a distribuição das médias bootstrap se aproxima
muito bem de uma normal (visível no histograma simétrico em `distribuicao_bootstrap.png`), o que
justifica a validade da aproximação paramétrica mesmo com a assimetria de base pronunciada na
variável individual.

---

## 8. Teste de Hipóteses e Teste A/B (`ab_testing.py`)

**Cenário (tema Saúde Pública):** segmentação por **Alta Cobertura Vacinal** (Grupo A:
`people_vaccinated_per_hundred` acima da mediana da amostra) vs. **Baixa Cobertura** (Grupo B:
igual ou abaixo da mediana), comparando o ritmo diário de vacinação (`daily_vaccinations`).

> Nota: a mediana da coluna de corte é 0,0 (57% dos registros são zero, refletindo países/dias
> anteriores ao início da campanha de vacinação local). Por isso o corte usa desigualdade estrita
> (`> mediana` para o Grupo A), evitando que o Grupo B fique vazio quando há massa de observações
> concentrada exatamente no valor de corte.

**Hipóteses formais:**
- H0: μ_A = μ_B (não há diferença no ritmo médio de vacinação diária entre países/dias de alta e
  baixa cobertura vacinal acumulada).
- H1: μ_A ≠ μ_B (teste bicaudal).
- α = 0,05.

**Resultados** (2.000 permutações, embaralhamento de rótulos):

| Métrica | Valor |
|---|---|
| n Grupo A / Grupo B | 31.178 / 42.138 |
| X̄_A − X̄_B (observado) | 10.963,08 |
| Valor-p empírico (bicaudal) | 0,00050 |

Gráfico: [`plots/distribuicao_permutacao.png`](plots/distribuicao_permutacao.png).

**Conclusão formal:** como p-valor (0,00050) < α (0,05), **rejeita-se H0**. Há evidência
estatística de diferença no ritmo diário de vacinação entre os dois grupos.

**Significado prático:** países/dias que já acumularam maior cobertura vacinal (Grupo A) também
apresentam ritmo diário de aplicação de doses significativamente mais alto do que os de baixa
cobertura — coerente com uma dinâmica de "bola de neve" (mais infraestrutura logística já
instalada tende a manter um fluxo maior de doses aplicadas por dia). Para o domínio de negócio
(planejamento de saúde pública), isso sugere que o gargalo inicial de uma campanha de vacinação
(baixa cobertura, baixo ritmo) tende a se autorreforçar, reforçando a importância de investimento
concentrado nas fases iniciais da campanha em regiões que ainda não decolaram.

---

## 9. Modelagem Preditiva Supervisionada

### 9.1 Regressão Linear Múltipla (`regression.py`)

Variável resposta (Y): `total_vaccinations`. Preditoras: `daily_vaccinations` (X1) e
`people_fully_vaccinated` (X2).

| Coeficiente | Valor | Interpretação (Ceteris Paribus) |
|---|---|---|
| β0 (intercepto) | −13.461,19 | valor de referência do modelo quando X1 = X2 = 0 (sem significado prático isolado, apenas ajuste da reta) |
| β1 (`daily_vaccinations`) | 3,8885 | mantendo `people_fully_vaccinated` constante, cada dose aplicada a mais no dia está associada a um aumento de ≈3,89 no total acumulado de vacinações |
| β2 (`people_fully_vaccinated`) | 2,4138 | mantendo `daily_vaccinations` constante, cada pessoa a mais com esquema vacinal completo está associada a um aumento de ≈2,41 no total acumulado de vacinações |

**Qualidade do ajuste:** R² = 0,9651 (as duas preditoras explicam ~96,5% da variância do total
acumulado de vacinações) e RMSE ≈ 2.240.558,85 doses — um erro absoluto grande em termos
nominais, mas esperado dada a escala de `total_vaccinations` (que chega a centenas de milhões em
países populosos).

### 9.2 Classificação Binária (`machine_learning.py`)

**Tarefa:** classificar cada registro como "vacinação rápida" (`is_fast_vax = 1`, quando
`daily_vaccinations_per_million` está acima da mediana) ou não, a partir das taxas acumuladas
(`total_vaccinations_per_hundred`, `people_vaccinated_per_hundred`,
`people_fully_vaccinated_per_hundred`).

Pipeline SKLearn com `StandardScaler` + `GridSearchCV` (cv=5, otimizando F1-Score):

| Modelo | Melhor Hiperparâmetro | Acurácia | Precisão | Recall | F1-Score |
|---|---|---|---|---|---|
| Regressão Logística | C = 10,0 | 0,6431 | 0,7297 | 0,4515 | 0,5578 |
| KNN | k = 3 | 0,7212 | 0,9179 | 0,4841 | 0,6339 |

O KNN superou a Regressão Logística em todas as métricas no conjunto de teste, sugerindo que a
fronteira de decisão entre "vacinação rápida" e "lenta" não é bem aproximada por uma combinação
linear das taxas acumuladas — há não-linearidades locais que o KNN, ao usar vizinhança no espaço
padronizado, capta melhor. O recall moderado (≈0,48–0,45) em ambos os modelos indica que ainda há
uma fração relevante de casos de "vacinação rápida" não identificados, sinalizando espaço para
features adicionais (ex.: variação temporal da taxa, e não apenas o nível acumulado).

---

## 10. Aprendizado Não Supervisionado e Redução de Dimensionalidade (`unsupervised.py`)

**PCA:** aplicado sobre as quatro variáveis numéricas padronizadas
(`total_vaccinations_per_hundred`, `people_vaccinated_per_hundred`,
`people_fully_vaccinated_per_hundred`, `daily_vaccinations_per_million`).

- Variância explicada por PC1: 72,98%
- Variância explicada por PC2: 23,96%
- **Variância acumulada (PC1 + PC2): 96,94%**

Como os dois primeiros componentes já capturam quase toda a variância dos dados, o plano PC1×PC2
(`plots/pca_projecao.png`) é uma representação fiel da estrutura original de 4 dimensões, com perda
de informação desprezível (~3%). Isso é esperado: as quatro variáveis originais são fortemente
correlacionadas entre si (todas medem, sob ângulos distintos, "quanto uma região avançou na
campanha de vacinação"), então poucos componentes bastam para resumir o mesmo sinal subjacente.

**K-Means:** com k = 3 (escolhido a partir do cotovelo em `plots/curva_cotovelo_kmeans.png`, onde a
queda de inércia se torna marginal a partir de k ≈ 3), os clusters resultantes têm os tamanhos:

| Cluster | Tamanho | Interpretação física |
|---|---|---|
| 1 | 50.538 | Estágio inicial/baixo — países-dia com baixa cobertura acumulada e baixo ritmo diário (early-stage da campanha) |
| 0 | 14.177 | Estágio intermediário — cobertura e ritmo moderados |
| 2 | 8.601 | Estágio avançado — alta cobertura acumulada e/ou ritmo elevado (campanhas maduras) |

Gráfico: [`plots/clusters_kmeans.png`](plots/clusters_kmeans.png). Na prática, os clusters funcionam como um
proxy de "estágio da campanha de vacinação" de cada país em cada data, útil para segmentar
políticas públicas (ex.: reforço logístico prioritário nos países ainda no cluster de estágio
inicial).

---

## 11. Inferência Causal e Tomada de Decisão

### a) Correlação vs. causalidade nos modelos supervisionados

Os coeficientes da regressão (Seção 9.1) e o poder preditivo do KNN/Regressão Logística (Seção
9.2) mostram **associações estatísticas fortes**, não relações causais comprovadas. Por
construção, `total_vaccinations`, `people_fully_vaccinated` e `daily_vaccinations` são
mutuamente dependentes por definição contábil (o total acumulado é, em grande parte, a soma dos
valores diários) — ou seja, parte da correlação observada na regressão é **mecânica/tautológica**,
não um efeito causal de uma variável independente sobre outra. O mesmo vale para a classificação:
"estar no estágio avançado da campanha" (alto `people_vaccinated_per_hundred`) e "ter ritmo diário
alto" (`daily_vaccinations_per_million`) tendem a covariar por estarem ambos ligados ao mesmo
processo temporal subjacente (a curva de vacinação do país), não porque um necessariamente causa o
outro.

### b) Potenciais variáveis de confusão (confounders)

1. **Tempo desde o início da campanha em cada país:** países que começaram a vacinar mais cedo
   acumulam naturalmente `total_vaccinations` e `people_fully_vaccinated` maiores, e isso
   confunde qualquer leitura causal entre as variáveis preditoras e a resposta — a "causa" real
   pode ser simplesmente "quantos dias de campanha já se passaram".
2. **Capacidade logística/orçamentária do país** (mencionada também na Parte 1): influencia
   simultaneamente o ritmo diário de aplicação (`daily_vaccinations`) e a velocidade com que a
   cobertura acumulada cresce, violando a hipótese de que as preditoras da regressão são
   exógenas/independentes do erro — um país com boa logística tem ambas as variáveis elevadas por
   um motivo comum, não por uma influenciar causalmente a outra.
3. **Disponibilidade de imunizante (oferta global/contratos):** afeta diretamente o ritmo diário
   de vacinação de forma exógena ao comportamento do próprio país, mas não está representada em
   nenhuma das features usadas nos modelos — um confounder omitido que pode inflar a aparente
   relação entre as taxas acumuladas e o rótulo de "vacinação rápida" na classificação.

### c) Tomada de decisão operacional

Com base no comportamento dos clusters (Seção 10) e nas previsões de classificação (Seção 9.2),
propõe-se: **priorizar reforço logístico e campanhas de conscientização nos países/datas
classificados no cluster de estágio inicial (baixa cobertura acumulada e baixo ritmo diário)**,
usando o classificador KNN (maior precisão, 0,918) como filtro operacional para sinalizar, em
tempo real, quais países-dia estão performando abaixo do esperado dado seu nível de cobertura
atual — permitindo direcionar recursos (doses, equipes móveis, comunicação) para os casos que o
modelo aponta como estagnados, em vez de distribuir o reforço de forma uniforme entre todos os
países.