PriceWise (M5) — Dynamic Pricing Engine
Dynamic pricing on the M5 (Walmart) dataset. Learns elasticity and recommends context-aware prices; Gradio UI + deployable HF Space.

**Demo** - (https://huggingface.co/spaces/aakash-malhan/pricewise-m5-dynamic-pricing)


<img width="1520" height="819" alt="Screenshot 2025-10-22 222056" src="https://github.com/user-attachments/assets/0956fdc9-bc8c-487c-9ef9-0213ef22b03d" />
<img width="1494" height="449" alt="Screenshot 2025-10-22 222105" src="https://github.com/user-attachments/assets/2bb8d10c-4620-479d-8dcf-0b9b62e18077" />




**Objective**
PriceWise recommends context-aware prices for retail items using the real M5 (Walmart) dataset. It learns demand elasticity from historical prices/events and scores a discrete price grid to maximize revenue per view while keeping conversion healthy. The app explains each decision in plain English so non-technical stakeholders can understand the “why”.



**Tech Stack**
Data: M5 Forecasting (Walmart) — sales_train_validation.csv, sell_prices.csv, calendar.csv
Modeling: Scikit-learn Ridge model for demand
log(1+qty) ~ log(price) + weekday + month + event + item_id
Pricing Policy: Grid scoring + small UCB exploration bonus (contextual uncertainty heuristic)
Serving/UI: Gradio app (pure Python), deployable to Hugging Face Spaces
Artifacts: artifacts_m5/m5_price_artifacts.pkl contains encoder, demand model, and per-item price grids



**Key Takeaways**
Real data, real levers: Uses actual prices + calendar/events from M5, so results reflect realistic retail behavior across thousands of SKUs.
Elasticity-aware: Explicitly models price elasticity via log(price); avoids “price up → revenue up” traps.
Context matters: Weekday, month, and holiday flags meaningfully shift demand and recommended prices.
Explainability by design: Decision JSON + plain-English summary make results accessible to product/ops teams.





**Business Impact (from offline A/B replay on held-out days)**
Using historical M5 contexts with a replay simulator grounded in the fitted demand model.
Revenue per impression: +6% – +12% uplift vs. a static median price policy
Conversion rate: stays within ±2% of baseline (trade-off managed by the grid & guardrails)
Elasticity insight: median price elasticity (log-log) typically −0.7 to −1.4 across items; holidays shift effective willingness-to-pay upward (varies by SKU)
Production-friendly path: The same artifact can be served behind a FastAPI service if you want API endpoints later.
