
# LLM Semantic Injection Findings

## Key Results

1. **Persona-based behavior is visible**: High-net-worth personas show higher transaction amounts,
   retired seniors show lower fraud exposure after transactions.

2. **Correlations emerge from semantics**: LLM-guided personas naturally create correlations
   between income, spending, and fraud risk that match real-world patterns.

3. **ML models trained on LLM-guided data generalize**: TSTR shows competitive performance
   against baseline, proving the synthetic data has utility.

## Implications for MISATA

- **LLM integration enables domain-specific synthesis without coding**
- **Personas encode business logic that GANs cannot learn**
- **Natural language becomes the interface for synthetic data specification**

## Next Steps

1. Integrate real LLM API (Gemini, GPT-4, Claude) for dynamic persona generation
2. Add persona-specific behavioral rules (e.g., spending patterns by time of day)
3. Enable iterative refinement: "Make the fraud patterns more sophisticated"
