import os, math, json
import numpy as np
import pandas as pd
import joblib
import gradio as gr

# ---------- Locate artifact (env -> common paths) ----------
ARTIFACT = os.getenv("M5_ARTIFACT")
if not ARTIFACT:
    for cand in [
        "artifacts_m5/m5_price_artifacts.pkl",  # preferred folder layout
        "m5_price_artifacts.pkl",               # root layout
    ]:
        if os.path.exists(cand):
            ARTIFACT = cand
            break

if not ARTIFACT or not os.path.exists(ARTIFACT):
    def _fatal_ui():
        msg = (
            "❌ Could not find model artifact.\n\n"
            "Looked for:\n"
            " - artifacts_m5/m5_price_artifacts.pkl\n"
            " - m5_price_artifacts.pkl\n\n"
            f"Working dir: {os.getcwd()}\n"
            f"Files here: {os.listdir('.')}"
        )
        with gr.Blocks() as demo:
            gr.Markdown(f"### {msg}")
        demo.launch()
    _fatal_ui()
    raise FileNotFoundError("Model artifact missing.")

# ---------- Load trained pieces ----------
blob        = joblib.load(ARTIFACT)
enc         = blob["onehot"]                 # OneHotEncoder for context
feat_cols   = blob["feature_cols"]           # ['log_price','weekday','month','is_event','item_id']
ctx_cols    = blob["ctx_cols"]               # ['weekday','month','is_event']
price_grid  = blob["price_grid"]             # dict[item_id] -> [prices]
model_pipe  = blob["model_pipe"]             # Ridge pipeline predicting log(1+qty)

# Prepare dropdown choices
ITEMS = [k for k, v in price_grid.items() if isinstance(v, (list, tuple)) and len(v) >= 2]
ITEMS = sorted(ITEMS)[:1500]  # cap for UI responsiveness
WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ---------- Helper functions ----------
def expected_qty(item_id: str, weekday: str, month: int, is_event: int, trial_price: float) -> float:
    row = {
        "log_price": math.log(trial_price),
        "weekday":   weekday,
        "month":     int(month),
        "is_event":  int(is_event),
        "item_id":   item_id
    }
    mu = float(model_pipe.predict(pd.DataFrame([row]))[0])  # predicts log(1+qty)
    return max(0.0, math.exp(mu) - 1.0)

def expected_revenue(item_id: str, weekday: str, month: int, is_event: int, trial_price: float) -> float:
    return float(trial_price * expected_qty(item_id, weekday, month, is_event, trial_price))

def choose_price(item_id, weekday, month, is_event, min_margin, explore):
    if not item_id:
        return gr.update(value=""), pd.DataFrame(), "Pick an item_id.", ""

    grid = price_grid.get(item_id, [])
    if not grid or len(grid) < 2:
        return gr.update(value=""), pd.DataFrame(), f"No usable price grid for {item_id}.", ""

    # Optional margin guardrail
    mm = None
    if isinstance(min_margin, str):
        min_margin = min_margin.strip()
    if min_margin not in (None, "", " "):
        try:
            mm = float(min_margin)
        except Exception:
            return gr.update(value=""), pd.DataFrame(), "min_margin must be numeric.", ""

    # Simple exploration bonus based on context encoding dispersion (UI-friendly heuristic)
    X = enc.transform(pd.DataFrame([{"weekday":weekday, "month":int(month), "is_event":int(is_event)}]))
    sigma = float(np.sqrt((X * (1 - X)).sum()))  # quick uncertainty proxy
    alpha = 0.8 if explore else 0.0

    rows, scores = [], []
    penalized = False
    for p in grid:
        if mm is not None and p < mm:
            penalized = True
            rows.append(dict(price=p, exp_qty=0.0, exp_rev=-1e9, ucb=-1e9))
            scores.append(-1e9)
            continue
        er  = expected_revenue(item_id, weekday, int(month), int(is_event), p)
        qty = expected_qty(item_id, weekday, int(month), int(is_event), p)
        ucb = er + alpha * sigma
        rows.append(dict(price=p, exp_qty=qty, exp_rev=er, ucb=ucb))
        scores.append(ucb)

    best_idx   = int(np.argmax(scores))
    best_price = float(grid[best_idx])
    best_row   = [r for r in rows if r["price"] == best_price][0]
    table = pd.DataFrame(rows).sort_values("price").reset_index(drop=True)

    # JSON for power users
    pretty = {
        "artifact_path": ARTIFACT,
        "item_id": item_id,
        "context": {"weekday": weekday, "month": int(month), "is_event": int(is_event)},
        "grid": [float(x) for x in grid],
        "chosen_price": best_price,
        "arm_index": best_idx,
        "scores_ucb": [float(s) for s in scores]
    }

    # ---- Plain-English summary for recruiters ----
    parts = []
    parts.append(
        f"**Recommended price:** **${best_price:,.2f}** for item **{item_id}** "
        f"on **{weekday} (month {int(month)})**"
        f"{' with a holiday/event' if int(is_event)==1 else ''}."
    )
    parts.append(
        f"This choice maximizes expected *revenue per view* on the tested grid by balancing "
        f"price vs. predicted demand. At ${best_price:,.2f}, the model expects about "
        f"**{best_row['exp_qty']:.2f} units** and **${best_row['exp_rev']:.2f} revenue**."
    )
    if penalized and mm is not None:
        parts.append(
            f"Prices below the **min margin ${mm:,.2f}** were **blocked** by a guardrail."
        )
    if alpha > 0:
        parts.append(
            "An exploration bonus (UCB) slightly prefers prices where the model is less certain, "
            "helping discover better options over time."
        )
    parts.append(
        "Behind the scenes, the app uses a demand model trained on the **M5 (Walmart)** dataset "
        "— `log(1+qty) ~ log(price) + weekday + month + event + item_id` — to estimate elasticity "
        "and score each price in the grid."
    )
    summary_md = " ".join(parts)

    note = f"✅ Recommended price: {best_price:.2f}"
    return json.dumps(pretty, indent=2), table, note, summary_md

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("# PriceWise (M5) — Dynamic Pricing Demo")
    gr.Markdown(
        "Pick an **item_id** and context. The app scores a discrete price grid using a "
        "demand model trained on the M5 dataset and a small exploration bonus."
    )

    with gr.Row():
        item = gr.Dropdown(choices=ITEMS, label="item_id (M5)", interactive=True)
        weekday = gr.Dropdown(WEEKDAYS, value="Friday", label="Weekday")
        month = gr.Slider(1, 12, value=11, step=1, label="Month")
        is_event = gr.Checkbox(label="Holiday/Event")

    with gr.Row():
        min_margin = gr.Textbox(label="Min margin ($, optional)", placeholder="e.g., 7.50")
        explore = gr.Checkbox(value=True, label="Explore (UCB bonus)")
        btn = gr.Button("Recommend Price", variant="primary")

    out_json = gr.Code(label="Decision JSON")
    out_tbl = gr.Dataframe(label="Grid scores (expected qty & revenue)", interactive=False)
    note = gr.Markdown("")
    human = gr.Markdown("")   # <-- new human-readable summary

    btn.click(
        choose_price,
        inputs=[item, weekday, month, is_event, min_margin, explore],
        outputs=[out_json, out_tbl, note, human]
    )

demo.launch()
