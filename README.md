# A Unified Taxonomy for Investigating the Mechanisms of Multilingual Multimodal Misinformation

This repository accompanies the paper **"A Unified Taxonomy for Investigating the Mechanisms of Multilingual MultiModal Misinformation"** ([arXiv:2608.29681](https://arxiv.org/abs/2608.29681)).

 Multimodal misinformation on social media is highly prevalent, potent, and harmful, yet difficult to detect and counter, and still poorly understood compared to its text-only counterpart. Research on the properties and deceptive strategies of multimodal misinformation is hindered by a lack of taxonomies grounded in real-world contexts and by the limitations of current multimodal machine learning models, which prevent the automation of annotation and analysis at scale. We address these shortcomings in three steps. First, we collect a large-scale, high-quality dataset of real-world misinformation instances from Twitter/X in seven languages. Second, we develop a novel, comprehensive taxonomy of multimodal misinformation grounded in an in-depth qualitative analysis of the data and prior theoretical work. Finally, we operationalise the taxonomy through an automated multi-step annotation pipeline using a Vision-Language Model (VLM), and perform human-validation. Our novel approach leads to previously undocumented insights about how social media users combine images with text to spread misinformation in the wild, e.g., that AI-generated content is particularly prevalent in technology and science, while vaccination misinformation disproportionately utilises images from news outlets to assert credibility. Our method and findings provide guidance for targeted approaches for detecting multimodal misinformation, and suggest that mitigation efforts should be developed and applied strategically rather than uniformly.

 This repository releases **the prompts driving that pipeline** and **the resulting machine-annotated dataset**.

## Repository contents

```
prompts/
  CommunityNotes/     Prompts for the X/Twitter Community Notes corpus
  ammeba/             Prompts for the fact-check-verdict corpus (AMMeBa-style)
data/
  merged_predictions_just_tweet_ids.csv   Released annotations (28,881 posts)
```

## The released dataset

`data/merged_predictions_just_tweet_ids.csv` contains **28,881 rows**, one per annotated post from the Community Notes corpus (28,830 distinct images).

### Privacy

The public release deliberately omits identifiable and copyrighted material. It contains **no user names, no tweet text, and no images** — posts are referenced only by `tweet_id`, and images only by `image_name`. The Community Note attached to each post is included. If you need the full version for research purposes, please contact us.

### Columns

| Column | Description |
| --- | --- |
| `tweet_id` | X/Twitter status ID of the annotated post. |
| `note` | The Community Note attached to the post — the `ADDITIONAL_CONTEXT` given to the annotator. |
| `image_name` | Filename of the post's image (not released; used to join with the full data). |
| `image_type` | Multi-label list of visual-form categories (e.g. `SIMPLE_PHOTO_PEOPLE`, `SOCIAL_MEDIA_SCREENSHOT`, `MEME`). |
| `classification` | What kind of problematic content the post is: `misinformation`, `ad`, `scam`, `stolen_content`, `engagement_bait`. |
| `emotion` | Multi-label list of emotions the poster appears to be evoking with the image. |
| `topic` | Multi-label topic list (`politics`, `health`, `conflict`, `sports`, …). |
| `message` | One-sentence paraphrase of the post's intended message, stated as if it were true. |
| `message_abstract_level_1..3` | Progressively more abstract paraphrases of `message`, for narrative-level clustering. |
| `strategy` | Multi-label rhetorical role of the image (Marsh & White-style codes, see below). |
| `image_misleads` | `YES`/`NO` — whether the *image* contributes to the deception, or is merely illustrative. |
| `mechanism` | Coarse mechanism by which the image misleads (only when `image_misleads == YES`). |
| `sub_mechanism` | Fine-grained mechanism, conditioned on `mechanism` (only for the five mechanisms that have children). |
| `*_reasoning` | Free-text justification the model produced alongside each corresponding label. |

Multi-label fields are stored as string representations of Python lists, so parse them with `ast.literal_eval` (see the snippet below).

### Label spaces

**`classification`** — `misinformation` (26,566), `stolen_content` (762), `ad` (664), `engagement_bait` (637), `scam` (202).

**`image_type`** (multi-label) — `SIMPLE_PHOTO_{PEOPLE,OBJECT,EVENT,ENVIRONMENT,DOCUMENT,OTHER}`, `SOCIAL_MEDIA_SCREENSHOT`, `NEWS_SCREENSHOT`, `OTHER_SCREENSHOT`, `OFFICIAL_DOCUMENT`, `OTHER_TEXT`, `DATA_VISUALIZATION`, `DIGITAL_ART`, `GRAPHIC_DESIGN`, `MEME`, `ANNOTATED_IMAGE`, `CAPTIONED_IMAGE`, `IMAGE_COLLAGE`, `COMPLEX_COMPOSITE`, `OTHER`.

**`strategy`** (multi-label) — `A1_decorate`, `A2_elicit_emotion`, `A3_control`, `B1_reiterate`, `B2_organize`, `B3_relate`, `B4_condense`, `B5_explain`, `C1_interpret`, `C2_develop`, `C3_transform`.

**`mechanism` → `sub_mechanism`** — the deception hierarchy, applied only where `image_misleads == YES`:

| Mechanism | Sub-mechanisms |
| --- | --- |
| `slanted` (8,083) | `exaggeration`, `misinterpretation_of_relevance`, `scientific_errors_or_conspiracies`, `other_slanted_representation` |
| `mismatch` (4,931) | `identity_mismatch`, `time_mismatch`, `place_mismatch`, `event_mismatch`, `other_mismatch` |
| `textual_claim_image` (2,622) | — |
| `fake_image` (2,502) | `AI_generated`, `forgery`, `staged`, `other_mechanism` |
| `manipulated_image` (2,454) | `addition`, `removal`, `replacement`, `textual`, `other_edit` |
| `unreliable_source` (463) | `satire`, `imposter`, `low_credibility_source`, `other_provenance_issues` |
| `other_mechanism` (269) | — |
| `nothing` (102) | — |
| `deny_authenticity` (55) | — |

`mechanism` is empty for the 7,355 posts labelled `image_misleads == NO` (plus a small number of parse failures), and `sub_mechanism` is additionally empty for mechanisms with no children.

### Loading

```python
import ast
import pandas as pd

df = pd.read_csv("data/merged_predictions_just_tweet_ids.csv")

for col in ["image_type", "emotion", "topic", "strategy"]:
    df[col] = df[col].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else [])

misleading = df[df.image_misleads == "YES"]
print(misleading.mechanism.value_counts())
```

## The prompts

`prompts/` holds the full text of every prompt used to build the dataset, in two parallel sets:

- **`CommunityNotes/`** — for social-media posts paired with a crowd-written Community Note. The prompts refer to `POST`, `IMAGE`, `USERNAME`, and `ADDITIONAL_CONTEXT`.
- **`ammeba/`** — the same taxonomy applied to fact-checked claims, where the context is a professional verdict rather than a community note. The prompts refer to `CLAIM`, `IMAGE`, `CLAIMANT`, and `FACT_CHECK_VERDICT`.

The two sets are otherwise near-identical, so a diff between them shows exactly what had to change to port the taxonomy to a different corpus.

Each file has the same structure: a `<SYSTEM_PROMPT>` block, a `<USER_PROMPT>` block containing the taxonomy definition, few-shot examples and a strict JSON output schema, and `{{placeholder}}` slots filled at inference time (`{{username}}`, `{{post}}`, `{{image}}`, `{{message}}`, `{{reasoning}}`, `{{mechanism}}`, `{{sub_mechanism_options}}`, `{{examples}}`). Every prompt asks the model for a label **and** a `reasoning` field, both of which are preserved in the released CSV.

| Prompt | Produces |
| --- | --- |
| `image_type.txt` | `image_type` — visual form of the image (image only, no text context). |
| `classification.txt` | `classification` — nature of the problematic content, chosen in priority order. |
| `topic_and_message.txt` | `topic`, `message`, `message_abstract_level_1..3`. |
| `rhetoric_role.txt` | `strategy` — how the image acts on the reader's understanding, conditioned on `message`. |
| `emotion.txt` | `emotion` — what the *poster* is trying to evoke, not what a reader happens to feel. |
| `multimodal_mechanism.txt` | Stage 1: `topic`, `message`, and the `image_misleads` gate. |
| `multimodal_mechanism_stage_2.txt` | Stage 2: `mechanism`, conditioned on stage 1's reasoning. |
| `multimodal_mechanism_stage_3.txt` | Stage 3: `sub_mechanism`, conditioned on the chosen mechanism, with mechanism-specific few-shot examples. |





## Citation

```bibtex
@misc{borenstein2026mmmmmunifiedtaxonomyinvestigating,
      title={MMMMM: A Unified Taxonomy for Investigating the Mechanisms of Multilingual MultiModal Misinformation}, 
      author={Nadav Borenstein and Greta Warren and Desmond Elliott and Isabelle Augenstein},
      year={2026},
      eprint={2608.29681},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2608.29681}, 
}
```

## Contact

For the full dataset (including tweet text, user names, and images), or with any question about the taxonomy or the annotation pipeline, please get in touch with the authors.
