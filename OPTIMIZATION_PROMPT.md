# TASK SPECIFICATION: ML Pipeline Optimization & Minimalist UI Overhaul

## ⚠️ AGENT EXECUTION RULES (STRICT STEP-BY-STEP MODE)
1. DO NOT implement the entire spec at once. 
2. Complete ONLY **Phase 1**, write/run validation scripts, and PAUSE.
3. Prompt the user to test the changes and create a Git commit before moving to the next phase.
4. Maintain modular code edits to prevent unnecessary credit consumption.

---

## Phase 1: Feature Engineering & Preprocessing Pipeline
**Objective:** Upgrade data processing scripts without touching the API or UI.

### Tasks
- **Outlier Cleaning:** Filter out training records where `GrLivArea > 4000` AND `SalePrice < 300000`.
- **Feature Aggregation:** Add transformed domain features:
  - `TotalSF`: `GrLivArea + TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
  - `TotalBaths`: `FullBath + (0.5 * HalfBath) + BsmtFullBath + (0.5 * BsmtHalfBath)`
  - `HouseAge`: `YrSold - YearBuilt`
  - `IsRemodeled`: `1 if YearRemodAdd != YearBuilt else 0`
  - `QualityScore`: `OverallQual * OverallCond`
- **Target Log Transformation:** Ensure training targets use `np.log1p(SalePrice)`.

### Verification & Git Checkpoint
- Run unit/pipeline tests to confirm transformed output shapes.
- **Stop Execution** and output: 
  > *"Phase 1 Complete. Please test, run `git add .` and `git commit -m 'feat: optimize feature engineering and log targets'`, then tell me to start Phase 2."*

---

## Phase 2: FastAPI Model Ensembling & API Refactoring
**Objective:** Update model training logic and inference endpoints.

### Tasks
- **Rescaling at Inference:** Ensure endpoint applies `np.expm1(y_pred)` to convert log predictions back to raw USD.
- **Model Stacking:** Implement weighted ensemble prediction:
  - Final Prediction Weight = 50% XGBoost + 30% LightGBM + 20% Ridge
- **Response Schema:** Return individual model estimates alongside the final ensemble prediction and feature importance rankings.

### Verification & Git Checkpoint
- Test the endpoint locally via `pytest` or curl requests to confirm valid JSON responses.
- **Stop Execution** and output:
  > *"Phase 2 Complete. Please test the backend, run `git add .` and `git commit -m 'feat: add stacked model ensemble and log scaling endpoints'`, then tell me to start Phase 3."*

---

## Phase 3: Minimalist UI Redesign (Streamlit)
**Objective:** Overhaul dashboard visual design and user flow.

### Design System (Zinc / Dark Slate)
- **Primary Accent:** Indigo (`#6366f1`)
- **Background / Containers:** Slate (`#0f172a` bg, `#1e293b` cards, `#334155` borders)
- **Typography:** Inter / System Sans-Serif (`#f8fafc` text, `#94a3b8` secondary text)

### Components & Layout
- **Feature Presets:** Add a dropdown (`Custom`, `Starter Home`, `Suburban Average`, `Luxury Estate`). Auto-sync sub-qualities (`KitchenQual`, `ExterQual`) with `OverallQual` to prevent feature drag.
- **Top Row KPI:** Hero `$ Price` display card with 95% confidence interval badge and price/sq ft metrics.
- **Output Section:** Minimalist Plotly dark-mode horizontal bar chart for feature importances.

### Verification & Git Checkpoint
- Launch Streamlit locally to verify layout responsiveness and error-free UI interaction.
- **Stop Execution** and output:
  > *"Phase 3 Complete. Please review the dashboard, run `git add .` and `git commit -m 'style: redesign Streamlit dashboard with zinc dark theme and feature presets'`."*