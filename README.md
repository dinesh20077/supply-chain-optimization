# supply-chain-optimization & revenue optimization ai

An AI-powered decision-intelligence platform that integrates supply-chain and business data to detect risks, predict demand, simulate disruptions, and optimize decisions.

Current Status: All 5 Modules Implemented & Verified (136/136 tests passing)

Quick Start
Prerequisites
Python 3.10+
PostgreSQL 14+ (running locally or accessible)
1. Clone & Setup
cd sciro
python -m venv .venv
.venv\Scripts\activate        # Windows

pip install -r requirements.txt





2. Configure Environment
copy .env.example .env
# Edit .env with your PostgreSQL credentials
Your .env file should contain:

DB_HOST=localhost
DB_PORT=5432
DB_NAME=sciro_db
DB_USER=postgres
DB_PASSWORD=your_password
LOG_LEVEL=INFO
DATA_DIR=data








3. Create the Database
Create the sciro_db database in PostgreSQL:

CREATE DATABASE sciro_db;
4. Run Database Migration
python -m sciro.database.migrate
This creates all tables, indexes, constraints, triggers, and analytics views in the sciro schema.






5. Generate Sample Data
python -m sciro.scripts.generate_sample_data
Generates realistic synthetic CSV files in data/sample/ for all entities:

10 regions (Indian geography)
50 suppliers (with varying reliability)
200 products (8 categories)
~500+ inventory records (with stockout scenarios)
1,000 shipments (with delay patterns)
300 customers (3 tiers)
5,000 orders (with seasonality)
~15,000 order items
6. Load Data into Database
python -m sciro.ingestion.pipeline
Loads all CSV files from data/sample/ into PostgreSQL in FK-dependency order.

7. Run Tests
python -m pytest tests/ -v

1. Data & Database	✅ Complete	Schema, ingestion, ETL, validation
2. Supply Chain & Business Intelligence	✅ Complete	Analytics engine, risk scoring, revenue-at-risk
3. AI/ML Prediction Engine	✅ Complete	Demand, revenue, stockout, supplier risk forecasting
4. Simulation & Optimization	✅ Complete	Disruption simulation, pricing, inventory allocation
5. Security, AI Copilot & Dashboard	✅ Complete	Auth, RBAC, AI assistant, Streamlit dashboard
Currency
All monetary values are in INR (₹).

License
This project is for educational purposes.
