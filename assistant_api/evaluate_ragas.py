"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.

Эталонные ответы (ground_truth) заданы вручную под фрагменты ГК / URDG в корпусе —
так Context Precision оценивает ретрив относительно смысла, а не обрезки ответа модели.
Context Utilization — насколько извлечённый контекст полезен для фактического ответа RAG.
"""

import math
import os
import sys
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from rag_pipeline import RAGPipeline

# Метрики и run_config (RAGAS 0.2+)
try:
    from ragas.metrics._context_precision import ContextPrecision, ContextUtilization
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    _METRICS_MODE = "v2"
except ImportError:
    _METRICS_MODE = "legacy"
    try:
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        from ragas.metrics import faithfulness, context_precision


# Вопросы по независимым гарантиям (под корпус ГК / URDG / обзор ВС)
EVALUATION_QUESTIONS = [
    "В какой срок гарант должен рассмотреть требование бенефициара по независимой гарантии?",
    "Чем независимая гарантия отличается от поручительства, если гарантию выдало лицо, не указанное в ГК как уполномоченное?",
    "Может ли гарант отказать бенефициару со ссылкой на нарушения по основному договору?",
    "Что такое «надлежащее представление» в смысле URDG?",
    "Какие основания для отказа в удовлетворении требования по независимой гарантии предусмотрены в ГК РФ?",
]

# Эталонные формулировки по текстам из базы (не дословная цитата, но смысл для метрик с reference)
EVALUATION_GROUND_TRUTHS = [
    "По ГК РФ гарант рассматривает требование бенефициара в течение пяти дней со дня, следующего за днём получения требования со всеми приложёнными документами; иным сроком гарантии может быть установлен иной срок, не превышающий тридцати дней (ст. 375 ГК РФ).",
    "Независимые гарантии выдают банки, иные кредитные организации и иные коммерческие организации; к обязательствам лиц, не указанных среди уполномоченных, применяются правила о договоре поручительства (п. 3 ст. 368 ГК РФ).",
    "Гарант не вправе ссылаться на нарушения по основному договору против требования бенефициара (п. 2 ст. 370 ГК РФ); отказ допускается по основаниям, в том числе из ст. 376 ГК РФ (несоответствие требования условиям гарантии, просрочка и др.).",
    "По URDG надлежащее представление по гарантии — это представление, соответствующее условиям гарантии, правилам URDG в согласующейся части и при отсутствии таких положений — международной стандартной практике; надлежащее требование удовлетворяет критериям надлежащего представления (ст. 2 URDG).",
    "Гарант отказывает при несоответствии требования или документов условиям гарантии либо при подаче по истечении срока гарантии; возможно приостановление платежа до семи дней по основаниям п. 2 ст. 376 ГК РФ; перечень оснований в целом исчерпывающий для отказа (ст. 376 ГК РФ).",
]


def _build_metrics():
    if _METRICS_MODE == "v2":
        return [
            Faithfulness(max_retries=3),
            ContextPrecision(max_retries=3),
            ContextUtilization(max_retries=3),
        ], RunConfig(timeout=180, max_retries=10, max_workers=8)
    return [faithfulness(), context_precision()], None


def _metric_keys(metrics):
    if _METRICS_MODE == "v2":
        return ["faithfulness", "context_precision", "context_utilization"]
    return ["faithfulness", "context_precision"]


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    if len(questions) != len(EVALUATION_GROUND_TRUTHS):
        raise ValueError("Число вопросов и эталонных ответов не совпадает")

    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("[*] Получение ответов от RAG системы...\n")

    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")

        result = pipeline.query(question, use_cache=False)

        questions_list.append(question)
        answers_list.append(result["answer"])
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        ground_truths_list.append(EVALUATION_GROUND_TRUTHS[i - 1])

        print("     [+] Ответ получен от OpenAI API")

    print()

    return Dataset.from_dict(
        {
            "question": questions_list,
            "answer": answers_list,
            "contexts": contexts_list,
            "ground_truth": ground_truths_list,
        }
    )


def _print_metric_block(title: str, result, key: str):
    values = [v for v in result[key] if not (isinstance(v, float) and math.isnan(v))]
    if values:
        avg = sum(values) / len(values)
        print(f"   {title}: {avg:.4f}")
    else:
        print(f"   {title}: нет валидных значений")


def _print_per_question(result, keys, questions):
    for i, question in enumerate(questions):
        print(f"\n{i + 1}. {question}")
        for key in keys:
            label = {
                "faithfulness": "Faithfulness",
                "context_precision": "Context precision (к эталону)",
                "context_utilization": "Context utilization (к ответу RAG)",
            }.get(key, key)
            val = result[key][i]
            if not (isinstance(val, float) and math.isnan(val)):
                print(f"   {label}: {val:.4f}")
            else:
                print(f"   {label}: не удалось вычислить")


def evaluate_rag_system():
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        sys.exit(1)

    metrics_to_use, run_config = _build_metrics()

    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"),
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)

    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("=" * 70)

    print("\n[*] Запуск оценки метрик RAGAS...")
    if _METRICS_MODE == "v2":
        print("   Метрики: Faithfulness, Context precision (с эталоном), Context utilization (с ответом RAG)")
    else:
        print("   Метрики: Faithfulness, Context precision (legacy-импорт)")
    print("   (несколько минут: вызовы LLM для метрик)\n")

    eval_kw = {"dataset": dataset, "metrics": metrics_to_use}
    if run_config is not None:
        eval_kw["run_config"] = run_config

    try:
        result = evaluate(**eval_kw)
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    keys = _metric_keys(metrics_to_use)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70 + "\n")

    _print_metric_block("Faithfulness (ответ vs контекст)", result, "faithfulness")
    _print_metric_block("Context precision (контекст vs эталон)", result, "context_precision")
    if "context_utilization" in keys:
        _print_metric_block("Context utilization (контекст vs ответ RAG)", result, "context_utilization")

    numeric_avgs = []
    for k in keys:
        vals = [v for v in result[k] if not (isinstance(v, float) and math.isnan(v))]
        if vals:
            numeric_avgs.append(sum(vals) / len(vals))
    avg_score = sum(numeric_avgs) / len(numeric_avgs) if numeric_avgs else 0.0

    print(f"\n{'─' * 70}")
    print(f"[ИТОГО] Среднее по метрикам: {avg_score:.4f}")

    if avg_score >= 0.7:
        print("   Оценка: высокие показатели [OK]")
    elif avg_score >= 0.5:
        print("   Оценка: удовлетворительно [!]")
    else:
        print("   Оценка: есть запас по ретриву/промпту [X]")

    print("\n" + "=" * 70)
    print("ДЕТАЛЬНО ПО ВОПРОСАМ")
    print("=" * 70)
    _print_per_question(result, keys, EVALUATION_QUESTIONS)

    print("\n" + "=" * 70)
    print("[INFO] КАК ЧИТАТЬ МЕТРИКИ")
    print("=" * 70)
    print("""
Faithfulness — насколько ответ RAG выводим из переданного контекста (без «галлюцинаций»).

Context precision (с эталоном) — насколько каждый извлечённый фрагмент полезен для ответа,
согласующегося с заранее заданным эталоном (EVALUATION_GROUND_TRUTHS). Раньше вместо эталона
использовалась обрезка ответа модели, из‑за чего оценка искажалась.

Context utilization — то же по смыслу, но опорный «ответ» — фактический ответ вашей RAG;
удобно, когда ответ хороший, а эталон формулирован иначе.

Faithfulness «не удалось вычислить» — чаще всего LLM RAGAS не разбил ответ на тезисы (пустой список);
повторный запуск или другая модель для метрик может помочь.
    """)

    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()
