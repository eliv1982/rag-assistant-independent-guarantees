"""
Компактный офлайн-бенчмарк извлечения: вопросы, написанные вручную, и ожидаемые нормы/якоря.

Гоняется весь настоящий путь индексации и поиска: нарезка -> VectorStore.load_corpus -> Chroma ->
VectorStore.search -> дедупликация -> шлюз достаточности данных. Подменены только эмбеддинги
(детерминированный лексический TF-IDF из fake_embeddings.py), поэтому сеть и OpenAI не используются.

Что это измеряет: находится ли в выдаче нужная статья, часть, пункт формы или позиция обзора, то есть
качество нарезки, метаданных и привязки. Что это НЕ измеряет: качество настоящей модели эмбеддингов.
Расстояния здесь — расстояния лексической заглушки, их нельзя переносить на text-embedding-3-small.

Тест-регрессия, а не метрика для витрины: для каждого случая зафиксирован допустимый ранг, пороги
hit@k — нижние границы по текущему поведению.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vector_store
from chunking import ChunkingConfig, build_chunks
from corpus_config import default_corpus_entries
from evidence import assess_evidence, collection_metric
from fake_embeddings import FakeEmbeddingsClient, HashingTfidfEmbedder
from retrieval_utils import deduplicate_context_docs

TOP_K = 5  # как RAG_TOP_K по умолчанию
CANDIDATE_K = max(TOP_K * 3, TOP_K + 5)  # как в RAGPipeline.query

# Порог шлюза для ЛЕКСИЧЕСКОЙ заглушки. Не порог боевой системы: у настоящих эмбеддингов шкала другая.
# Взят серединой между наблюдаемыми диапазонами: худший лучший фрагмент среди вопросов по теме корпуса
# 0.772, лучший фрагмент посторонних вопросов 0.922 -> (0.772 + 0.922) / 2 ~ 0.85.
# Запас с обеих сторон проверяет test_gate_margin_is_visible.
FAKE_EMBEDDING_MAX_DISTANCE = 0.85

GK = "gk_rf_368_379"
FZ44 = "fz_44_art_45"
FZ223 = "fz_223_art_3_4"
PP1005 = "pp_1005_2013"
PP1397 = "pp_1397_2022"
VS2019 = "vs_independent_guarantee_2019"
VS2017 = "vs_contract_system_2017"


def anchor(source, **fields):
    return dict(source=source, **fields)


# (id, вопрос, допустимые якоря (любой из них засчитывается), максимальный ранг в выдаче)
POSITIVE_CASES = [
    (
        "gk_general_rules",
        "Что такое независимая гарантия и какие организации могут выдавать независимые гарантии?",
        [anchor(GK, article="368")],
        1,
    ),
    (
        "gk_independence",
        "Зависит ли обязательство гаранта перед бенефициаром от основного обязательства?",
        # то же правило излагает позиция 11 Обзора ВС РФ 2019: она законно конкурирует со ст. 370 ГК РФ
        [anchor(GK, article="370"), anchor(VS2019, position=11)],
        1,
    ),
    (
        "gk_demand_timing",
        "В какой срок бенефициар должен представить требование по независимой гарантии?",
        [anchor(GK, article="374")],
        2,
    ),
    (
        "gk_refusal",
        "В каких случаях гарант отказывает бенефициару в удовлетворении требования?",
        [anchor(GK, article="376", point="1")],
        2,
    ),
    (
        "gk_suspension",
        "На какой срок гарант вправе приостановить платеж и по каким основаниям?",
        [anchor(GK, article="376", point="2")],
        3,
    ),
    (
        "fz44_requirements",
        "Какие условия обязательно должна содержать независимая гарантия для заказчика: "
        "безотзывность, сумма, неустойка, срок действия?",
        [anchor(FZ44, article="45", part="2")],
        1,
    ),
    (
        "fz44_refusal_to_accept",
        "Каковы основания для отказа заказчика в принятии независимой гарантии?",
        [anchor(FZ44, article="45", part="6")],
        2,
    ),
    (
        "fz44_register",
        "В какой срок гарант должен внести независимую гарантию в реестр и направить принципалу выписку из реестра?",
        # ч. 8 (выписка в течение рабочего дня) и ч. 11 (внесение не позднее рабочего дня); то же в Правилах ПП №1005
        [
            anchor(FZ44, article="45", part="8"),
            anchor(FZ44, article="45", part="11"),
            anchor(PP1005, anchor="registry_rules/p.5"),
        ],
        2,
    ),
    (
        "fz223_sme_bid_guarantee",
        "Каким требованиям должна соответствовать независимая гарантия в обеспечение заявки "
        "в закупке с участием субъектов малого и среднего предпринимательства?",
        [anchor(FZ223, article="3.4", part="14.1")],
        1,
    ),
    (
        "fz223_penalty",
        "Какую неустойку уплачивает гарант за просрочку по независимой гарантии "
        "в закупках с участием субъектов малого и среднего предпринимательства?",
        [anchor(FZ223, article="3.4", part="14.3")],
        1,
    ),
    (
        "pp1005_contract_form_payment",
        "В течение скольких рабочих дней гарант обязан уплатить бенефициару денежную сумму по требованию "
        "согласно типовой форме независимой гарантии в обеспечение исполнения контракта?",
        [anchor(PP1005, form="contract_guarantee_form", point="10")],
        1,
    ),
    (
        "pp1005_registry_number",
        "Какую структуру имеет уникальный номер реестровой записи в реестре независимых гарантий?",
        [anchor(PP1005, anchor="registry_rules/p.11"), anchor(PP1005, anchor="closed_registry_rules/p.15")],
        1,
    ),
    (
        "pp1397_documents_with_demand",
        "Какие документы заказчик представляет гаранту одновременно с требованием об уплате "
        "по независимой гарантии в обеспечение заявки в закупках МСП?",
        [
            anchor(PP1397, anchor="regulation/p.8"),
            anchor(PP1397, form="bid_guarantee_form", point="7"),
            anchor(PP1397, form="bid_demand_form", point="3"),
        ],
        2,
    ),
    (
        "pp1397_registry_features",
        "Особенности ведения реестра независимых гарантий для целей Закона о закупках",
        [anchor(PP1397, anchor="regulation/p.10")],
        2,
    ),
    (
        "vs2019_position_7",
        "Недействительность соглашения о выдаче гарантии между принципалом и гарантом "
        "является ли основанием для отказа бенефициару?",
        [anchor(VS2019, position=7)],
        1,
    ),
    (
        "vs2019_position_14",
        "Является ли банкротство гаранта основанием для прекращения обязательств по гарантии?",
        [anchor(VS2019, position=14)],
        1,
    ),
    (
        "vs2019_position_13",
        "Расходы принципала на оплату банковской гарантии по прекращенному муниципальному контракту: "
        "кто возмещает?",
        [anchor(VS2019, position=13)],
        1,
    ),
    (
        "vs2019_position_17",
        "Считается ли соблюденным досудебный порядок при направлении бенефициаром требования о платеже?",
        [anchor(VS2019, position=17)],
        1,
    ),
    (
        "vs2017_position_25",
        "Банковская гарантия не соответствует требованиям закона: можно ли признать победителя торгов "
        "уклонившимся от заключения контракта?",
        [anchor(VS2017, position=25)],
        1,
    ),
    (
        "vs2017_position_30",
        "Получил ли заказчик деньги по банковской гарантии: вправе ли исполнитель требовать возмещения убытков?",
        [anchor(VS2017, position=30)],
        1,
    ),
]

OUT_OF_SCOPE_CASES = [
    ("borscht", "Как приготовить борщ в домашних условиях?"),
    ("weather", "Какая погода будет завтра в Москве?"),
    ("flight", "Сколько стоит билет на самолет из Москвы в Сочи?"),
    ("football", "Кто стал чемпионом мира по футболу?"),
]


def _matches(meta, expectation):
    return all(meta.get(key) == value for key, value in expectation.items())


def rank_of(docs, expectations):
    """Ранг (с 1) первого фрагмента, подходящего под любой из допустимых якорей; None, если нет."""
    for rank, doc in enumerate(docs, start=1):
        if any(_matches(doc["metadata"], e) for e in expectations):
            return rank
    return None


class Retrieval:
    """Индекс всего корпуса в настоящем Chroma с детерминированными эмбеддингами."""

    def __init__(self, directory):
        config = ChunkingConfig(800, 200, 80)
        texts = []
        for entry in default_corpus_entries():
            chunks = build_chunks(
                Path(entry["path"]).read_text(encoding="utf-8"),
                source=entry["source"],
                source_display=entry["source_display"],
                source_kind=entry["source_kind"],
                doc_type=entry["doc_type"],
                chunker=entry["chunker"],
                citation=entry["citation"],
                config=config,
            )
            texts.extend(c.text for c in chunks)
        self.embedder = HashingTfidfEmbedder().fit(texts)
        self.client = FakeEmbeddingsClient(self.embedder)
        env = {"RAG_CHUNK_SIZE": "800", "RAG_CHUNK_OVERLAP": "200", "RAG_MIN_CHUNK_LEN": "80", "ANONYMIZED_TELEMETRY": "False"}
        with mock.patch.dict(os.environ, env), mock.patch.object(vector_store, "get_openai_client", return_value=self.client):
            with contextlib.redirect_stdout(io.StringIO()):
                self.store = vector_store.VectorStore(collection_name="benchmark", persist_directory=directory)
                self.store.load_corpus(default_corpus_entries())
        self.metric = collection_metric(self.store.collection)
        self.chunk_count = self.store.collection.count()

    def search(self, question):
        raw = self.store.search(question, top_k=CANDIDATE_K)
        return raw, deduplicate_context_docs(raw, max_docs=TOP_K)


@contextlib.contextmanager
def retrieval():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        yield Retrieval(tmp)


def evaluate():
    """Все случаи: [(id, ранг, лучшее косинусное расстояние)], [(id, лучшее расстояние, достаточно ли данных)]."""
    with retrieval() as r:
        positives = []
        for case_id, question, expectations, _ in POSITIVE_CASES:
            raw, context = r.search(question)
            assessment = assess_evidence(raw, r.metric, FAKE_EMBEDDING_MAX_DISTANCE)
            positives.append((case_id, rank_of(context, expectations), assessment.best_distance, assessment.sufficient))
        negatives = []
        for case_id, question in OUT_OF_SCOPE_CASES:
            raw, _ = r.search(question)
            assessment = assess_evidence(raw, r.metric, FAKE_EMBEDDING_MAX_DISTANCE)
            negatives.append((case_id, assessment.best_distance, assessment.sufficient))
        return r.chunk_count, r.metric, positives, negatives


def summarize(positives):
    total = len(positives)
    return {
        "hit@1": sum(1 for _, rank, _, _ in positives if rank == 1) / total,
        "hit@3": sum(1 for _, rank, _, _ in positives if rank is not None and rank <= 3) / total,
        "hit@5": sum(1 for _, rank, _, _ in positives if rank is not None and rank <= TOP_K) / total,
    }


# Нижние границы по текущему поведению (измерено: hit@1 0.65, hit@3 1.0, hit@5 1.0); запас на безвредные
# изменения нарезки. hit@5 (то, что попадает в промпт) обязан быть полным.
HIT_AT_1_FLOOR = 0.6
HIT_AT_3_FLOOR = 0.95


class TestRetrievalBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunk_count, cls.metric, cls.positives, cls.negatives = evaluate()
        cls.summary = summarize(cls.positives)

    def test_benchmark_is_compact_and_covers_the_corpus(self):
        self.assertGreaterEqual(len(POSITIVE_CASES) + len(OUT_OF_SCOPE_CASES), 12)
        self.assertLessEqual(len(POSITIVE_CASES) + len(OUT_OF_SCOPE_CASES), 24)
        self.assertGreaterEqual(len(OUT_OF_SCOPE_CASES), 2)
        self.assertLessEqual(len(OUT_OF_SCOPE_CASES), 4)
        sources = {a["source"] for _, _, anchors, _ in POSITIVE_CASES for a in anchors}
        self.assertEqual(sources, {GK, FZ44, FZ223, PP1005, PP1397, VS2019, VS2017})

    def test_index_uses_cosine_metric_read_from_the_collection(self):
        self.assertEqual(self.metric, "cosine")
        self.assertGreater(self.chunk_count, 300)

    def test_positive_cases_hit_the_expected_source_and_anchor(self):
        pinned = {case_id: max_rank for case_id, _, _, max_rank in POSITIVE_CASES}
        for case_id, rank, distance, _ in self.positives:
            self.assertIsNotNone(rank, f"{case_id}: ожидаемый якорь не найден в топ-{TOP_K}")
            self.assertLessEqual(rank, pinned[case_id], f"{case_id}: ранг {rank} хуже зафиксированного {pinned[case_id]}")

    def test_aggregate_hit_rates_do_not_regress(self):
        print(f"\nretrieval benchmark: {self.summary}")
        self.assertEqual(self.summary["hit@5"], 1.0)
        self.assertGreaterEqual(self.summary["hit@3"], HIT_AT_3_FLOOR)
        self.assertGreaterEqual(self.summary["hit@1"], HIT_AT_1_FLOOR)

    def test_positive_cases_pass_the_evidence_gate(self):
        for case_id, rank, distance, sufficient in self.positives:
            self.assertTrue(sufficient, f"{case_id}: ложный отказ, лучшее расстояние {distance}")

    def test_out_of_scope_cases_trigger_the_no_evidence_path(self):
        for case_id, distance, sufficient in self.negatives:
            self.assertFalse(sufficient, f"{case_id}: посторонний вопрос прошёл шлюз, расстояние {distance}")

    def test_gate_margin_is_visible(self):
        worst_positive = max(d for _, _, d, _ in self.positives)
        best_negative = min(d for _, d, _ in self.negatives)
        self.assertLess(worst_positive, FAKE_EMBEDDING_MAX_DISTANCE)
        self.assertGreater(best_negative, FAKE_EMBEDDING_MAX_DISTANCE)


if __name__ == "__main__":
    count, metric, positives, negatives = evaluate()
    print(f"chunks={count} metric={metric}")
    for case_id, rank, distance, sufficient in positives:
        print(f"  + {case_id:34s} rank={rank} best_distance={distance:.3f} gate_ok={sufficient}")
    for case_id, distance, sufficient in negatives:
        print(f"  - {case_id:34s} best_distance={distance:.3f} gate_ok={sufficient}")
    print(summarize(positives))
    unittest.main(argv=[sys.argv[0]])
