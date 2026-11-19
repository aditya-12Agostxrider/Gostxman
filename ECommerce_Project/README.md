# E-Commerce Sales Analytics

Data analytics project on 5,000 orders and 25,000 customers across 7 countries.
Finds what's driving low customer ratings (2.97/5) and delivery delays using Python, SQL, and Power BI.

## Files
| File | Description |
|---|---|
| `data_cleaning.py` | Cleans raw data, creates star schema tables |
| `python_analytics.py` | EDA charts + statistical hypothesis tests |
| `predictive_modeling.py` | Revenue forecasting + K-Means customer clustering |
| `sql_analysis.sql` | 20+ SQL queries for business analysis |
| `dashboard.html` | Open in browser — interactive BI dashboard |
| `sample_Fact_Sales.csv` | 50 sample order records |
| `sample_Dim_Customer.csv` | 50 sample customer records |
| `Dim_Product.csv` | 4 product categories |

## Run
```bash
pip install -r requirements.txt
run.bat
```
Or manually:
```bash
python data_cleaning.py
python python_analytics.py
python predictive_modeling.py
```
Then open `dashboard.html` in your browser.

## Key Results
- Net Revenue: $5.11M | AOV: $1,022 | Active Customers: 989 / 25,000
- Delivery speed does NOT affect ratings (p = 0.21)
- Payment method DOES affect ratings (Chi-sq p = 0.007)
- 44% of orders use heavy discounts (21–35%) — $1.13M given away

## Author
Aditya
