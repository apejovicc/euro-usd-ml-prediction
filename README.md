# Komparativna analiza modela mašinskog učenja za predviđanje deviznog kursa

**Predviđanje binarnog smjera kretanja EUR/USD kursa**  
*Projekat za master rad · ETF UCG*

---

## O projektu

Projekat poredi pet modela mašinskog učenja za predviđanje dnevnog smjera kretanja kursa EUR/USD:

- logistička regresija;
- SVM;
- XGBoost;
- LightGBM;
- LSTM.

Ciljna varijabla je binarna:

- `1` — kurs raste narednog dana;
- `0` — kurs pada ili ne raste narednog dana.

Eksperiment koristi zajednički skup karakteristika i istu hronološku podjelu za sve modele. Pored poređenja tačnosti, uključuje optimizaciju hiperparametara, kontrolu curenja informacija, test stacionarnosti, bootstrap intervale pouzdanosti, McNemar test i analizu generalizacije kroz vremenske podperiode.

## Preduslovi

- Python 3.12;
- virtuelno okruženje (`venv` ili `conda`);
- internet konekcija za Yahoo Finance i FRED/ALFRED pozive;
- FRED API ključ.

FRED ključ možete besplatno preuzeti na [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

> **Bezbijednost:** API ključ se ne čuva u kodu niti u `config.yaml`. U notebook-u se unosi pomoću `getpass` i postoji samo tokom aktivne sesije.

## Instalacija

Iz korijena projekta pokrenite:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Projekat koristi biblioteke `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow`, `statsmodels`, `yfinance`, `fredapi`, `matplotlib`, `PyYAML` i `joblib`.

## Struktura projekta

```text
Analiza/
├── config.yaml                    # Jedini izvor parametara eksperimenta
├── run_experiment.py              # Pokretanje iz komandne linije
├── requirements.txt
├── src/
│   ├── utils.py                   # Konfiguracija, seed i putanje
│   ├── data_loader.py             # Yahoo Finance i FRED/ALFRED podaci
│   ├── features.py                # Karakteristike, cilj i ADF test
│   ├── split.py                   # Hronološka podjela i CV foldovi
│   ├── evaluation.py              # Baseline modeli, metrike i testovi
│   ├── pipelines.py               # Orkestracija kompletnog eksperimenta
│   └── models/
│       ├── sklearn_models.py       # LogReg, SVM, XGBoost i LightGBM
│       └── lstm_model.py           # LSTM arhitektura i optimizacija
├── notebooks/
│   └── main_experiment.ipynb      # Glavni korisnički interfejs
├── data/
│   ├── raw/                       # Preuzeti OHLC i makro podaci
│   └── processed/                 # Obrađeni skupovi sa karakteristikama
└── results/
    ├── metrics_<tag>.csv
    ├── predictions_<tag>.csv
    ├── generalization_<tag>.csv
    ├── optimization_comparison_<tag>.csv
    ├── stationarity_<tag>.csv
    ├── subperiod_accuracy_<tag>.csv
    ├── summary_<tag>.json
    └── figures/
```

## Pokretanje

### Jupyter notebook

Notebook je preporučeni način rada jer API ključ ostaje u memoriji sesije:

```powershell
jupyter notebook notebooks/main_experiment.ipynb
```

Ćelije treba izvršavati redom:

1. import biblioteka i ispis verzija;
2. učitavanje `config.yaml`;
3. unos FRED API ključa pomoću `getpass`;
4. pokretanje punog eksperimenta sa `use_short_period=False`;
5. pregled rezultata i statističkih testova;
6. pokretanje dvogodišnjeg eksperimenta sa `use_short_period=True`;
7. generisanje figura iz `results/summary_*.json` fajlova.

### Komandna linija

```powershell
python run_experiment.py --fred-key YOUR_KEY
python run_experiment.py --fred-key YOUR_KEY --short
```

Opcije:

| Opcija | Značenje |
|---|---|
| `--config` | Putanja do alternativnog YAML konfiguracionog fajla |
| `--fred-key` | FRED API ključ |
| `--short` | Pokretanje dvogodišnjeg eksperimenta |

Za maksimalnu zaštitu ključa preporučuje se notebook unos pomoću `getpass`, umjesto čuvanja ključa u komandnoj istoriji.

## Podaci i karakteristike

Konfiguracija traži sljedeće periode:

| Eksperiment | Traženi period | Oznaka |
|---|---|---|
| Primarni | `2003-01-01` – `2026-01-01` | `full_23y` |
| Kratki | `2024-01-01` – `2026-01-01` | `short_2y` |

Model koristi devet karakteristika:

`Return`, `SMA_ratio`, `Volatility_5D`, `Momentum_10D`, `Daily_Range`, `RSI_14`, `CPI_YoY`, `UNRATE_Diff` i `Rate_Differential_Change`.

Efektivni početak obrađenog skupa je kasniji od traženog perioda zbog dostupnosti EUR/USD podataka i zagrijavanja karakteristika koje koriste duže vremenske prozore, posebno `CPI_YoY`.

Makroekonomske serije se učitavaju iz FRED/ALFRED izvora. Za `CPIAUCSL` i `UNRATE` koristi se point-in-time rekonstrukcija kako bi se izbjeglo korišćenje podataka koji u trenutku predviđanja još nisu bili objavljeni.

## Metodologija

- Podaci se dijele hronološki na trening, validacioni i testni skup.
- Ne koristi se nasumično miješanje vremenskih opservacija.
- `gap=1` uklanja granične redove između skupova radi kontrole curenja informacija.
- Klasični modeli koriste `GridSearchCV` sa `TimeSeriesSplit` validacijom.
- XGBoost i LightGBM koriste early stopping na validacionom skupu.
- LSTM pretraga obuhvata dužinu sekvence, broj jedinica, broj slojeva, dropout, learning rate i batch size.
- LSTM se trenira kroz više stohastičkih pokretanja, a rezultati se agregiraju.
- ADF test se primjenjuje na svih devet karakteristika.
- Bootstrap se koristi za intervale pouzdanosti tačnosti.
- McNemar test poredi modele na istom testnom skupu.

## Konfiguracija

Svi ključni parametri nalaze se u [config.yaml](config.yaml):

| Sekcija | Šta određuje |
|---|---|
| `random_seed` | Reproduktivnost NumPy, scikit-learn i TensorFlow eksperimenata |
| `data` | Tražene periode i oznake eksperimenata |
| `fred` | FRED/ALFRED serije i postavke izvora |
| `split` | Hronološke proporcije, broj CV foldova i gap |
| `lstm` | Prostor pretrage, broj epoha i stohastička pokretanja |
| `classical_models` | Grid parametri, broj poslova i early stopping |
| `paths` | Lokacije ulaznih podataka, rezultata i figura |

## Izlazni rezultati

Za svaki eksperiment generišu se fajlovi sa oznakom `<tag>`:

| Fajl | Sadržaj |
|---|---|
| `metrics_<tag>.csv` | Accuracy, precision, recall, F1 i matrice konfuzije |
| `predictions_<tag>.csv` | Sve stvarne i predviđene klase na testnom skupu |
| `generalization_<tag>.csv` | Trening, validaciona i testna tačnost; generalizacioni jaz |
| `optimization_comparison_<tag>.csv` | Tačnost prije i poslije optimizacije |
| `stationarity_<tag>.csv` | Rezultati ADF testa |
| `subperiod_accuracy_<tag>.csv` | Tačnost kroz tri vremenska podperioda |
| `summary_<tag>.json` | Kompletan sažetak korišćen za tabele i figure |
| `figures/` | Generisane figure za analizu i master rad |

`summary_<tag>.json` sadrži:

```text
tag, dataset_summary, split_info, cv_fold_info,
best_hyperparameters, lstm_search_log, early_stopping_info,
model_selection, final_params_used, lstm_n_test_sequences,
results_table, statistical_tests, generalization_table,
optimization_comparison, n_models_improved,
subperiod_accuracy, stationarity_table
```

## Trajanje

Najviše vremena troše:

1. LSTM grid search: do 15 kombinacija i fabrička konfiguracija, do 30 epoha po kombinaciji;
2. finalni LSTM: pet stohastičkih pokretanja po 60 epoha i jedno baseline pokretanje;
3. `GridSearchCV` za četiri klasična modela kroz pet vremenskih foldova;
4. pokretanje oba eksperimenta uzastopno, čime se ukupno vrijeme približno udvostručuje.

Za brzu provjeru prvo pokrenite dvogodišnji eksperiment sa opcijom `--short`.

## Poznate napomene

- `fred.series_lag_days` je dio konfiguracije, ali se trenutno ne primjenjuje kao direktno pomjeranje serija. Kontrola vremenskog curenja zasniva se na ALFRED point-in-time rekonstrukciji.
- Paralelizaciju LSTM pretrage treba prvo testirati na kratkom eksperimentu, posebno na Windows sistemima.
- Efektivni period u obrađenom skupu može biti kraći od perioda navedenog u konfiguraciji zbog čišćenja podataka i potrebnog zagrijavanja karakteristika.

## Reproduktivnost

Svi modeli koriste zajednički `random_seed` iz `config.yaml`, istu hronološku podjelu i isti skup karakteristika. Rezultati se automatski čuvaju u `data/processed/` i `results/`, što omogućava provjeru tabela i figura bez ponovnog računanja cijelog eksperimenta.

---

**EUR/USD predviđanje · Master rad · ETF UCG**