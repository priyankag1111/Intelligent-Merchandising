# H&M Intelligent Assortment & Merchandising – Streamlit POC

Prototype built for an H&M CEO pitch.  
Uses **only** the official [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) Kaggle dataset.

## What the app demonstrates

| Tab | Analysis | Maps to original Opportunity |
|-----|----------|------------------------------|
| 1. Customer Segmentation | RFM + K-Means | Segmentation |
| 2. Demand Forecasting | Prophet hierarchical weekly forecasts | Demand prediction |
| 3. Trend Detection | Sales velocity (accelerating / declining) | Trend detection |
| 4. Recommendations | Item-item CF + popularity hybrid | Ranking / Visibility |
| 5. Product Analytics | Performance, TF-IDF similarity, market basket | Assortment |
| 6. Deploy instructions | How to put it on Streamlit Cloud / GitHub | — |

## Quick start (local)

```bash
# 1. Clone
git clone <your-repo-url>
cd hm_streamlit_app

# 2. Create environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add data
# Download from Kaggle and place these three files (CSV or .csv.zip) in the data/ folder:
#   - articles.csv
#   - customers.csv
#   - transactions_train.csv

# 4. Run
streamlit run app.py
```

## Data sampling

The full transaction file is ~32 M rows.  
The app automatically samples a configurable number of customers (default 25 k) so that interactive analysis stays fast on a laptop or free Streamlit Cloud instance.

## Project structure

```
hm_streamlit_app/
├── app.py                     # Main Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
├── data/                      # Place CSVs here (git-ignored)
│   ├── articles.csv
│   ├── customers.csv
│   └── transactions_train.csv
└── src/
    ├── data_loader.py         # Loading + sampling
    ├── segmentation.py        # RFM + K-Means
    ├── demand_forecast.py     # Prophet weekly forecasts + velocity
    ├── recommendations.py     # Popularity + item-item CF hybrid
    └── product_analytics.py   # Performance, TF-IDF similarity, basket
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (keep the large CSVs out of git – they are already in `.gitignore`).
2. Go to https://share.streamlit.io → **New app**.
3. Select the repository, branch, and set the main file path to `app.py`.
4. For data access you have three practical options:
   - Host a **sampled** version of the three CSVs in the repo (recommended for a public demo).
   - Download the files at runtime from a public URL (Google Drive direct link, S3, etc.).
   - Use Streamlit Secrets + a private storage location.

## Notes for the CEO demo

- All numbers are based on a **sample** of the real H&M transaction data.
- The recommendation engine is a lightweight hybrid (collaborative filtering + popularity) – production systems would add sequential models, two-tower nets, and real-time features.
- Demand forecasts use Facebook Prophet at product-type / garment-group level; size-level forecasts would require additional size attributes.
- The same code structure can later ingest real event streams (search, views, carts) once those data sources are connected.

## License & data usage

This code is provided for demonstration and educational purposes.  
The underlying H&M data remains subject to the original Kaggle competition rules.
