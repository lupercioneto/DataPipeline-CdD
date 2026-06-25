# DataPipeline-CdD

Pipeline reprodutível de dados aplicado ao tema **Saúde Pública (COVID-19)**, construído para
a Avaliação Prática Unificada (Parte 1) da disciplina de Ciência de Dados.

Dataset utilizado: [Country Vaccinations](https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress) (Kaggle).

---

* **Equipe:**
  * Francisco Lupercio 
  * Daniel Galvão
  * João Algusto

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
│   ├── visualize/
│   │   └── visualize.py        # gráfico de integridade visual
│   ├── logs/
│   │   └── log.py              # logger simples
│   └── main.py                 # orquestrador do pipeline
├── dados_limpos_final.csv      # saída final consolidada
├── requirements.txt
└── README.md
```

Para executar: `cd src && python main.py`

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

A imagem é exportada automaticamente para `dados_vacinacao_diaria.png` na raiz do projeto, a
partir da base já tratada (`dados_limpos_final.csv`).

---

## 6. Saída do Pipeline

Ao final da execução de `main.py`, o pipeline gera na raiz do projeto:

- `dados_limpos_final.csv` — base de dados final, tratada e consolidada;
- `dados_vacinacao_diaria.png` — gráfico de integridade visual.