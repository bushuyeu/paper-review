Critically evaluate the paper as if reviewing for a prestigious conference (e.g., NeurIPS  "Review Content" or ICML “Main Track Reviewer Form Instructions”). Your review should be actionable rather than descriptive. In other words, it should provide explicit instructions on improving the paper.
Strengths:
- The paper delivers a fully open video-language model stack (weights, data, and code) and explicitly avoids distilling from closed VLMs, which materially advances open reproducibility in a space dominated by proprietary systems.
- The introduction of nine new datasets spanning dense video captioning, long-form QA, open-vocabulary pointing, and tracking addresses known gaps in open video grounding data, and the paper documents the data collection interfaces and pipelines.
- Training design choices (token weighting, packing, and message-tree attention masks) are clearly motivated by long-sequence multimodal constraints and are supported by ablations.
- The evaluation covers a broad suite of academic benchmarks plus new Molmo2-specific captioning, counting, and pointing metrics, and includes a large human preference study.
- The results show strong competitiveness among open models and sometimes approach proprietary systems, especially on grounding-oriented tasks.
Weaknesses:
- The vision encoder remains closed (SigLIP 2) and closed text-only LLMs are used for data generation, which weakens the “fully open” claim and makes full reproducibility dependent on proprietary components.
- The captioning metric relies on LLM-as-judge scoring, which can introduce bias and may not be comparable across model families; there is limited analysis of its stability or correlation with human judgments.
- The paper reports strong aggregate benchmark numbers but provides limited analysis of failure modes beyond the appendix; the main text does not quantify how grounding errors vary with video length, object frequency, or motion complexity.
- Evaluation details for baselines (frame sampling, prompts, decoding) are noted as sometimes unavailable, which complicates fair comparison; the paper could do more to standardize or disclose these settings.
- Some limitations (degenerate pointing outputs, long-video grounding limits) are acknowledged, but concrete mitigation experiments are not shown.
Suggestions for Improvement:
- Provide an explicit, reproducible alternative to the closed SigLIP 2 encoder (even if lower-performing), and include an ablation to show the trade-off between openness and performance.
- Add a targeted human evaluation for grounding (pointing and tracking) to validate automatic metrics, and report inter-annotator agreement for the new datasets.
- Expand error analysis in the main text with quantitative breakdowns by video length, object scale, and motion type to identify where grounding fails.
- Release detailed prompts and settings for all LLM-based data generation steps, and include a sensitivity study showing the impact of different LLMs or prompt variants on data quality.
- Standardize evaluation protocols for baselines (frames, prompts, decoding) wherever possible, and publish any missing eval scripts to reduce uncertainty in comparisons.

Longer-version:
Summary
Molmo2 presents a family of open-weight video-language models that support video understanding and pixel-level grounding (pointing and tracking). The paper introduces nine new datasets for dense video captioning, long-form QA, video pointing, and tracking, collected without using closed VLMs. The training pipeline uses a three-stage schedule, packing and message-tree attention, and token weighting to handle long outputs. On a broad benchmark suite and new Molmo2 evaluations, the models are competitive with strong open baselines and sometimes approach proprietary systems, particularly for grounding tasks.
Claims and Evidence
The core claims are (1) Molmo2 is state-of-the-art among open models, (2) grounding capabilities are strong for video pointing/tracking, and (3) the data and training recipe are fully open. The benchmark table and Molmo2-specific evaluations support (1) and (2) at the aggregate level. However, (3) is weakened by the use of a closed image encoder and closed text-only LLMs for data generation. Evidence for grounding quality is primarily metric-based; the paper would benefit from more direct human evaluation and failure-mode breakdowns, especially for long videos and high-frequency objects where the authors report degenerate outputs.
Relation to Prior Works
The paper positions itself against proprietary systems and open models that either do not release data or rely on closed VLMs for synthetic data (e.g., LLaVA-Video, ShareGPT4Video). It extends prior work on image grounding (e.g., PixMo) into video grounding and points to advances in video LLMs and tracking (e.g., Ref-VOS, MeVis, TAP) as related areas. The contribution is well-situated, but the paper could more explicitly contrast its dataset construction with other open-data efforts and clarify which aspects are truly novel versus scaled-up extensions.
How well-versed are you with the literature related to this paper? I keep up with the literature in this area, particularly on video-language models and grounding datasets.
Other Aspects
Clarity is generally good, with strong figures and a transparent data-collection narrative. The main paper is dense, and some essential reproducibility details (prompts, baseline evaluation settings) are pushed to the appendix. The breadth of tasks is impressive, but more space in the main text should be allocated to diagnostic analyses and known failure modes.
Questions for Authors
1) Can you release an open alternative to SigLIP 2 and quantify the performance gap for Molmo2-4B/8B? This would clarify the dependency on closed visual encoders and its practical impact.  
2) What is the variance of your LLM-as-judge captioning metric across different judge models or prompt templates? If it varies substantially, how does that affect the ranking of open baselines?  
3) For video grounding, can you provide a breakdown of accuracy by video length and object motion complexity, and indicate whether the degenerate pointing failure mode is length-dependent?
Ethical Issues
I do not see immediate ethical issues that require an ethics review, but the paper should clarify dataset licensing and privacy safeguards for all video sources and annotations.
Code of Conduct Acknowledgement
I affirm the ICML code of conduct.
Overall Recommendation
4. Accept
