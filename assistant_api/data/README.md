# Legal corpus

Scope: independent guarantees under Russian law. The corpus combines the general Civil Code
framework with procurement-specific regulation (Federal Laws No. 44-FZ and No. 223-FZ and the
Government resolutions that implement them) and Supreme Court guidance on how these rules are
applied.

It is a **curated snapshot of seven sources**, not a complete body of law and not an automatic
legal-update feed. Read [Currentness and edition status](#currentness-and-edition-status) before
relying on any answer built from it.

## Sources

| # | File | Source | What is included | Source category | Retrieval role |
|---|------|--------|------------------|-----------------|----------------|
| 1 | `GK_368_379.txt` | Civil Code of the Russian Federation (Гражданский кодекс РФ), § 6 "Независимая гарантия" | Articles 368–379, including 375.1. Article 369 is a single status line (see below). | `law` | General rules of the independent guarantee (layer 1). Chunked by article, then paragraph. |
| 2 | `44FZ_article_45.txt` | Federal Law of 05.04.2013 No. 44-FZ "О контрактной системе в сфере закупок товаров, работ, услуг для обеспечения государственных и муниципальных нужд", Article 45 | Article 45, parts 1–13 (including 1.1–1.7, 3.1, 8.1, 8.2). No other article of 44-FZ. | `procurement_law` | Special procurement rules for guarantees under 44-FZ (layer 2). Chunked by article, part, item. |
| 3 | `223FZ_article_3_4_guarantees.txt` | Federal Law of 18.07.2011 No. 223-FZ "О закупках товаров, работ, услуг отдельными видами юридических лиц", Article 3.4 | Only the guarantee-related parts of Article 3.4: 12, 14.1, 14.2, 14.3, 17, 31, 32. The remaining parts of the article and the rest of 223-FZ are not included. | `procurement_law` | Special procurement rules for guarantees under 223-FZ, procurements open only to small and medium-sized businesses (layer 2). Chunked by article, part, item. |
| 4 | `PP_1005.txt` | Government Resolution of 08.11.2013 No. 1005 "О независимых гарантиях, используемых для целей Федерального закона «О контрактной системе…»" | The resolution and the documents it approves: additional requirements for the guarantee; list of documents the customer sends with the payment demand; rules for the register of guarantees in the unified procurement information system; payment-demand form; rules for the closed register; standard forms of bid-security and contract-performance guarantees. | `government_resolution` | Implementing requirements, registers and standard forms for 44-FZ (layer 3). Chunked by point, section, form. |
| 5 | `PP_1397.txt` | Government Resolution of 09.08.2022 No. 1397 "О независимых гарантиях, предоставляемых в качестве обеспечения заявки на участие в конкурентной закупке … с участием субъектов малого и среднего предпринимательства, и независимых гарантиях, предоставляемых в качестве обеспечения исполнения договора…" | The resolution, the Regulation (Положение) it approves, and Appendices 1–4 (two standard guarantee forms and two payment-demand forms). The amendments to other Government acts that the resolution also approves are not included. | `government_resolution` | Implementing requirements, forms and register rules for 223-FZ SME procurements (layer 3). Chunked by point, section, form. |
| 6 | `VS_independent_guarantee_2019.txt` | Review of judicial practice on disputes involving the legislation on independent guarantees, approved by the Presidium of the Supreme Court of the Russian Federation on 05.06.2019 | All 17 legal positions of the review. | `case_law_summary` | Interpretation and application of the rules (layer 4). Chunked by numbered position. |
| 7 | `VS_contract_system_2017_guarantees.txt` | Review of judicial practice on the application of the legislation on the contract system in procurement, approved by the Presidium of the Supreme Court of the Russian Federation on 28.06.2017 | Selected material only: the introductory part and positions 25 and 30, the two positions on independent (bank) guarantees. The rest of the review is not included. | `case_law_summary` | Interpretation and application in 44-FZ procurement (layer 4). Chunked by numbered position. |

Source metadata (category, citation label, chunking strategy) lives in
[`../corpus_config.py`](../corpus_config.py); the table above is the human-readable counterpart.
The listed contents were checked against the files in this directory. The `source_kind` values are
the project's own labels, not official classifications.

## Source layers

The assistant is instructed to identify the applicable layer and not to merge all sources
mechanically:

1. Civil Code: general independent-guarantee rules.
2. 44-FZ / 223-FZ: special procurement rules.
3. Government Resolutions No. 1005 / No. 1397: implementing requirements, registers and standard forms.
4. Supreme Court reviews: interpretation and application.

## Text policy

- The files contain legal source text, not generated paraphrases, summaries or third-party
  commentary. No external commentary has been added.
- Editorial amendment-history notes and unrelated amendment blocks may have been removed to reduce
  retrieval noise. A few such notes remain in place (for example in 44-FZ Article 45 and 223-FZ
  Article 3.4).
- Source structure is preserved: article and part numbers, point numbers, the numbering of legal
  positions and the sections of standard forms. Retrieval and citations rely on these anchors.
- Article 369 of the Civil Code is kept only as a single historical-status line (it lost force on
  1 June 2015 under Federal Law No. 42-FZ). No repealed substantive wording is included.
- Historical terminology is left as it appears in the sources. In particular, court materials keep
  the term "банковская гарантия" ("bank guarantee"), which the assistant treats as part of the
  independent-guarantee subject area.
- Changing any file changes `corpus_id`, so the index is rebuilt and cached answers (if the optional
  cache is enabled) are not reused.

## Currentness and edition status

- The corpus is a curated repository snapshot. It is **not** automatically synchronized with
  legislative changes, and nothing in the application checks whether a provision is still in force.
- The files do not record an official publication, a source database or a consolidated-edition
  date, so a "current as of" date is deliberately not stated here.
- Exact retrieval URLs and retrieval dates were not preserved for this curated snapshot; verify
  current versions against authoritative legal sources before use.
- The only date evidence inside the files is residual amendment notes. The latest amendment
  referenced in them is Federal Law of 16.04.2022 No. 109-FZ (in the 44-FZ and 223-FZ files).
  The files for the Civil Code, the Government resolutions and the Supreme Court reviews carry no
  edition marker. This shows what the text contained when it was saved; it does not show that no
  later amendment exists.
- Provisions that a fragment marks as repealed (for example Civil Code Article 369 and the repealed
  points of Resolution No. 1005) are included only as status lines. The prompt tells the model not
  to apply them as current law.
- Users must verify the current wording of the law against authoritative sources (official legal
  publication portals) before relying on an answer. The assistant's output is informational and is
  not legal advice.

## Rights

The project's software license ([`../../LICENSE`](../../LICENSE)) covers the project code and its
own documentation. The legal and court texts in this directory are separate source materials: they
are not original project code, and the project does not claim ownership of them or relicense them.
Anyone who redistributes or reuses these texts should check the terms that apply to the original
sources under applicable law.

## Changing the corpus

Edit or add a file, update the entry in [`../corpus_config.py`](../corpus_config.py) (and, for a new
document structure, the chunking strategy), and update this table. The application detects the new
`corpus_id` and builds a fresh Chroma collection on the next question; building it calls the
embeddings API.
