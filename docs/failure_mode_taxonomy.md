# Failure Mode Taxonomy
## LLM Evaluation and Monitoring System — Document Summarization

This document defines the five failure categories used by the automated evaluation pipeline. It was written before any code existed and serves as the design contract for the Gemini judge rubric. Each category describes a distinct failure mode, explains its production relevance, and specifies how the pipeline detects it.

---

### 1. Factual Consistency

**What it is:** The model output introduces a claim that contradicts information present in the source document. The claim is not invented from nothing — it is a distortion of something real in the source. For example, a source document states that operating income increased 12% year-over-year; the summary states it increased 21%. The number is plausible and domain-appropriate, but wrong.

**Why it matters in production:** Factual inconsistency is the failure mode most likely to cause downstream harm without being immediately obvious. A user reading a summary has no reason to doubt a figure that sounds reasonable. In financial contexts specifically, an inconsistent summary can influence decisions — investment theses, risk assessments, earnings interpretations — based on information the source document does not support. Unlike hallucination, factual inconsistency is harder to catch because the model is working with real information and getting it wrong in subtle ways, not fabricating wholesale.

**How the pipeline detects it:** The Gemini judge is given both the source document and the model output and asked to verify whether every factual claim in the output is supported by the source. It scores 0 (fail) if any claim in the output contradicts the source, 0.5 (marginal) if a claim is imprecisely stated but directionally correct, and 1 (pass) if all claims are accurately grounded in the source.

---

### 2. Completeness

**What it is:** The model output omits information that is materially important in the source document. The output may be factually accurate as far as it goes, but it leaves out content that a reasonable reader would need to form a correct understanding. Critically, completeness failures include omitting key supporting evidence, not just omitting primary conclusions — a summary that captures the headline result but drops the qualifying conditions or risk factors that shape its interpretation is a completeness failure.

**Why it matters in production:** Completeness failures are the most dangerous failure mode from a user trust perspective because they are nearly invisible. A factually inconsistent summary feels wrong to an attentive reader; an incomplete summary feels complete. In document summarization systems deployed at scale, systematic completeness failures erode the informational value of the system without producing any signal that something is wrong. Users make decisions on summaries they believe are representative and are not.

**How the pipeline detects it:** The Gemini judge evaluates whether the output captures the material substance of the source, defined as: the primary conclusions, the key evidence supporting those conclusions, and any significant qualifications or risk factors. It scores 0 if critical information is absent, 0.5 if secondary supporting detail is missing but the primary substance is present, and 1 if the output is materially complete relative to the source.

---

### 3. Coherence

**What it is:** The model output is internally inconsistent, logically contradictory, or structurally incoherent — independent of whether its claims are factually grounded in the source. A coherence failure occurs when the output contradicts itself, when claims are presented in an order that obscures their logical relationship, or when the output does not form a unified and readable whole. This category evaluates the output in isolation, not against the source.

**Why it matters in production:** Coherence is the hygiene check of the taxonomy. It catches a class of model failures that factual evaluation cannot — outputs that are individually accurate at the sentence level but fail at the structural level. In a production pipeline handling high document volume, incoherent outputs degrade user experience and signal model instability. Coherence scores also serve as a useful leading indicator: a drop in coherence across runs often precedes drops in factual quality and typically reflects model behavioral drift rather than input distribution shift.

**How the pipeline detects it:** The Gemini judge evaluates the output on its own terms — without reference to the source — and assesses whether it reads as a unified, logically structured response. It scores 0 if the output is self-contradictory or logically broken, 0.5 if the output is structurally awkward or partially incoherent but not misleading, and 1 if the output is internally consistent and well-structured.

---

### 4. Hallucination

**What it is:** The model output references entities, facts, figures, or events that do not appear anywhere in the source document and cannot be inferred from it. Unlike factual inconsistency — where the model distorts real information — hallucination is the introduction of content from outside the source entirely. Examples in the financial domain include referencing an executive not mentioned in the transcript, citing a product line not discussed, or stating a guidance figure that was never given.

**Why it matters in production:** Hallucination is the highest-severity failure mode in this taxonomy. It represents a fundamental breakdown in the model's grounding: the model is no longer summarizing the document, it is generating plausible-sounding content. In financial document summarization, hallucinated entities or figures are indistinguishable from real ones without consulting the source, which defeats the purpose of the summarization system entirely. Hallucination failures at production scale constitute a reliability failure of the system, not just a quality degradation.

**How the pipeline detects it:** The Gemini judge checks whether every entity, figure, date, and factual reference in the output has a traceable antecedent in the source document. Paraphrased or abstracted references count as grounded if the underlying referent is clearly present in the source. It scores 0 if any reference in the output has no basis in the source, 0.5 if a reference is loosely inferable but not explicitly stated, and 1 if all references are clearly grounded in the source.

---

### 5. Instruction Following

**What it is:** The model output fails to conform to the format, structure, or constraints specified in the prompt. This is a behavioral failure rather than a content failure — the model may produce accurate, complete, coherent, grounded content that nonetheless violates the output specification. Examples include producing a paragraph when a bullet list was requested, exceeding a specified length constraint, or omitting a required structural element.

**Why it matters in production:** Instruction following failures are operationally disruptive in ways that factual failures are not. Downstream systems that parse model outputs — dashboards, APIs, structured data pipelines — typically expect consistent output formats. A model that produces correct content in the wrong format can break downstream processing silently. Tracking instruction following as a distinct evaluation category makes format compliance a first-class metric, which is necessary for catching model updates or prompt changes that shift output structure.

**How the pipeline detects it:** The Gemini judge is given the original prompt specification alongside the output and evaluates whether the output conforms to all stated constraints. It scores 0 if the output violates a material format constraint, 0.5 if the output partially conforms but deviates in secondary ways, and 1 if the output fully conforms to the specified format and constraints.

---

## Severity Classification

Not all failure modes carry equal production risk. The taxonomy is ordered from highest to lowest severity:

| Rank | Category | Severity | Rationale |
|---|---|---|---|
| 1 | Hallucination | Critical | Content fabricated from outside the source; no user signal |
| 2 | Factual Consistency | High | Real information distorted; plausible and hard to detect |
| 3 | Completeness | High | Critical information absent; output appears complete |
| 4 | Instruction Following | Medium | Format violations break downstream systems |
| 5 | Coherence | Low-Medium | Structural failures; visible to attentive readers |

Regression alerts are weighted by severity: consecutive regressions in Hallucination or Factual Consistency trigger immediate SNS notification; regressions in Coherence or Instruction Following are logged and flagged in the dashboard but do not trigger immediate alerts unless sustained across three or more consecutive runs.

---
