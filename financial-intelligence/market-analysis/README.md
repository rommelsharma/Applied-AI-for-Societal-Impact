# Value Investor - Focus on Indian Markets

## About this project

This is a private GitHub project.

I will share the design motivation behind this project. Today there are a lot of apps in the market that you can subscribe to. There are a lot of open source or commercially available projects too to refer to for stock analysis. However I have been keen on value investing more than day trading or short term stock buy and sell. The inspiration comes from the works of Benjamin Graham and Peter Lynch documented in their books on investments.

## Why this design is different from existing projects

PKScreener is India's #1 open-source NSE screener but it's entirely built around momentum trading — RSI, MACD, breakout patterns. MachineLearningStocks uses scikit-learn to predict which stocks will outperform, but it's predicting price movements, not business value. Neither applies any investment philosophy. This solution treats Graham/Lynch rules as the *engine*, not decorations.

## The three things that make this solution unique

**1. The scoring engine is the philosophy, not a display layer.** Every stock gets a 0–100 score computed from actual Graham metrics (PE, P/B, D/E, dividend yield) and Lynch metrics (PEG, EPS CAGR, ROE). The PE exception is only granted if PEG < 1 — exactly as Lynch argued.

**2. The LLM doesn't just describe, it reasons.** Existing Streamlit+Ollama dashboards show technical indicators but lack fundamental reasoning. Here the LLM receives live fundamentals + news + your user thesis and applies the Graham/Lynch persona to produce a structured JSON verdict — agree, disagree, or augment your reasoning.

**3. The user prompt box respects human understanding and reasoning.** When you type *"I think Waaree Energies is a buy because India's solar capacity needs to triple..."*, the system fetches live data, cross-references recent news, and the LLM tells you whether your thesis holds up or where it's weak — using Graham/Lynch principles, not generic AI waffle.

## Technology choices rationale

Streamlit-based fundamental analysis cleanly implements filters for PE and PEG ratios — it's the right UI choice for this kind of data-heavy but non-frontend work. yfinance with the `.NS` suffix is the standard for fetching NSE fundamentals in Python, and nsetools fills the real-time gaps. For deep fundamentals like 5-year EPS CAGR (not in yfinance), we scrape screener.in.

---

> **Note:** There will not be any further details or files in the public folder as this project is personal and based on my beliefs in long term value investing, drawing on the learnings from Benjamin Graham and the likes. You are welcome to reach out for a discussion at a professional level.
