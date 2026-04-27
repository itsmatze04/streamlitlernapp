import random
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = [
    "question_text",
    "question_type",
    "topic",
    "difficulty",
    "explanation",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "correct_answers",
]

QUESTION_TYPE_ALIASES = {
    "single_choice": "single_choice",
    "single choice": "single_choice",
    "single": "single_choice",
    "multiple_choice": "multiple_choice",
    "multiple choice": "multiple_choice",
    "multiple": "multiple_choice",
    "multi": "multiple_choice",
    "true_false": "true_false",
    "true false": "true_false",
    "true/false": "true_false",
    "richtig/falsch": "true_false",
    "wahr/falsch": "true_false",
    "wahrfalsch": "true_false",
    "richtigfalsch": "true_false",
    "boolean": "true_false",
}

QUESTION_TYPE_LABELS = {
    "single_choice": "Single Choice",
    "multiple_choice": "Multiple Choice",
    "true_false": "Richtig/Falsch",
}


def normalize_text(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def load_questions(fallback_path: str = "data/questions.csv") -> pd.DataFrame:
    path = Path(fallback_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Standarddatei nicht gefunden: {fallback_path}. Bitte Datei im Repository anlegen."
        )
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise ValueError("Die Standard-CSV ist leer. Bitte Datei mit Header und Fragen befüllen.") from e


def validate_columns(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Fehlende Pflichtspalten: {', '.join(missing)}"
    return True, None


def normalize_question_type(value: object) -> str:
    cleaned = normalize_text(value).replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())
    mapped = QUESTION_TYPE_ALIASES.get(cleaned)
    if mapped:
        return mapped
    if "single" in cleaned:
        return "single_choice"
    if "multiple" in cleaned or "multi" in cleaned:
        return "multiple_choice"
    if any(term in cleaned for term in ["richtig", "falsch", "wahr", "unwahr", "true", "false"]):
        return "true_false"
    return "unknown"


def get_answer_options(row: pd.Series) -> List[Tuple[str, str]]:
    options: List[Tuple[str, str]] = []
    for key in ["a", "b", "c", "d"]:
        label = str(row.get(f"option_{key}", "")).strip()
        if label:
            options.append((key.upper(), label))

    if row.get("question_type") == "true_false":
        if len(options) < 2:
            return [("A", "Richtig"), ("B", "Falsch")]
        return options[:2]

    return options


def parse_correct_answers(raw: object, question_type: str, options: List[Tuple[str, str]]) -> Set[str]:
    text = normalize_text(raw)
    if not text:
        return set()

    normalized = text.replace(";", ",").replace("|", ",")
    tokens = [tok.strip() for tok in normalized.split(",") if tok.strip()]

    code_map: Dict[str, str] = {code.lower(): code for code, _ in options}
    label_map: Dict[str, str] = {normalize_text(label): code for code, label in options}
    truthy = {"richtig", "wahr", "true", "1", "ja"}
    falsy = {"falsch", "unwahr", "false", "0", "nein"}

    parsed: Set[str] = set()
    for token in tokens:
        t = normalize_text(token).replace("option_", "").replace("option", "").strip()

        if t in {"a", "b", "c", "d"}:
            parsed.add(t.upper())
            continue
        if t in code_map:
            parsed.add(code_map[t])
            continue
        if t in label_map:
            parsed.add(label_map[t])
            continue

        if question_type == "true_false":
            if t in truthy:
                parsed.add("A")
                continue
            if t in falsy:
                parsed.add("B")
                continue

    existing_codes = {code for code, _ in options}
    return {code for code in parsed if code in existing_codes}


def check_answer(user_answer: Set[str], correct_answer: Set[str], question_type: str) -> bool:
    if question_type == "multiple_choice":
        return set(user_answer) == set(correct_answer)
    return set(user_answer) == set(correct_answer)


def format_answer_set(answer_set: Set[str], options: List[Tuple[str, str]]) -> str:
    if not answer_set:
        return "Keine"
    lookup = {code: label for code, label in options}
    return ", ".join([f"{code}: {lookup.get(code, code)}" for code in sorted(answer_set)])


def sanitize_questions(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    out = df.copy()

    for col in REQUIRED_COLUMNS:
        out[col] = out[col].fillna("").astype(str).apply(lambda x: x.strip())

    out["question_type"] = out["question_type"].apply(normalize_question_type)
    unknown_mask = out["question_type"] == "unknown"
    if unknown_mask.any():
        warnings.append(f"{int(unknown_mask.sum())} Frage(n) mit unbekanntem question_type wurden übersprungen.")
        out = out[~unknown_mask].copy()

    valid_indices = []
    for idx, row in out.iterrows():
        options = get_answer_options(row)
        parsed = parse_correct_answers(row["correct_answers"], row["question_type"], options)
        if parsed:
            valid_indices.append(idx)
        else:
            warnings.append(f"Frage in CSV-Zeile {idx + 2} hat keine gültige richtige Antwort und wurde übersprungen.")

    out = out.loc[valid_indices].copy().reset_index(drop=True)
    return out, warnings


def init_state() -> None:
    defaults = {
        "learning_pointer": 0,
        "learn_checked": False,
        "exam_started": False,
        "exam_indices": [],
        "exam_pointer": 0,
        "exam_answers": {},
        "exam_finished": False,
        "pass_threshold": 60,
        "filter_signature": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_learning() -> None:
    st.session_state.learning_pointer = 0
    st.session_state.learn_checked = False


def reset_exam() -> None:
    st.session_state.exam_started = False
    st.session_state.exam_indices = []
    st.session_state.exam_pointer = 0
    st.session_state.exam_answers = {}
    st.session_state.exam_finished = False


def maybe_reset_on_context_change(signature: str) -> None:
    if st.session_state.filter_signature != signature:
        st.session_state.filter_signature = signature
        reset_learning()
        reset_exam()


def apply_filters(df: pd.DataFrame, topic: str, difficulty: str, qtype: str) -> pd.DataFrame:
    out = df.copy()
    if topic != "Alle":
        out = out[out["topic"] == topic]
    if difficulty != "Alle":
        out = out[out["difficulty"] == difficulty]
    if qtype != "Alle":
        out = out[out["question_type"] == qtype]
    return out


def render_header_stats(df: pd.DataFrame) -> None:
    topics = sorted([t for t in df["topic"].unique() if str(t).strip()])
    difficulties = sorted([d for d in df["difficulty"].unique() if str(d).strip()])
    qtypes = sorted(df["question_type"].unique())
    type_labels = [QUESTION_TYPE_LABELS.get(t, t) for t in qtypes]

    st.container(border=True).markdown(
        """
### Willkommen 👋
Trainiere Fragen im **Lernmodus** oder starte eine vollständige **Prüfungssimulation**.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Geladene Fragen", len(df))
    c2.metric("Themen", len(topics))
    c3.metric("Schwierigkeiten", len(difficulties))
    c4.metric("Fragetypen", len(type_labels))

    with st.expander("Verfügbare Inhalte anzeigen"):
        st.write(f"**Themen:** {', '.join(topics) if topics else '-'}")
        st.write(f"**Schwierigkeitsgrade:** {', '.join(difficulties) if difficulties else '-'}")
        st.write(f"**Fragetypen:** {', '.join(type_labels) if type_labels else '-'}")


def render_answer_widget(row: pd.Series, options: List[Tuple[str, str]], key_prefix: str) -> Set[str]:
    q_type = row["question_type"]
    if not options:
        st.warning("Diese Frage hat keine auswählbaren Optionen.")
        return set()

    labels = [f"{code}: {label}" for code, label in options]
    if q_type == "single_choice":
        selected = st.radio("Antwort auswählen", labels, key=f"{key_prefix}_single")
        return {selected.split(":", 1)[0]}
    if q_type == "multiple_choice":
        st.markdown("**Mehrfachauswahl:** Bitte die passenden Antworten anklicken.")
        selected_codes: Set[str] = set()
        for code, label in options:
            checked = st.checkbox(
                f"{code}: {label}",
                key=f"{key_prefix}_multi_{code}",
                help="Klicke auf das Kästchen oder direkt auf den Antworttext.",
            )
            if checked:
                selected_codes.add(code)
        return selected_codes

    selected = st.radio("Richtig oder Falsch?", labels[:2], key=f"{key_prefix}_tf")
    return {selected.split(":", 1)[0]}


def render_learning_mode(df: pd.DataFrame, random_order: bool) -> None:
    st.header("Lernmodus")
    if df.empty:
        st.info("Keine Fragen nach aktuellem Filter verfügbar.")
        return

    indices = list(df.index)
    if random_order:
        random.Random(42).shuffle(indices)

    ptr = min(st.session_state.learning_pointer, len(indices) - 1)
    st.session_state.learning_pointer = ptr

    idx = indices[ptr]
    row = df.loc[idx]
    options = get_answer_options(row)
    correct = parse_correct_answers(row["correct_answers"], row["question_type"], options)

    st.progress((ptr + 1) / len(indices), text=f"Frage {ptr + 1} von {len(indices)}")
    with st.container(border=True):
        st.subheader(row["question_text"])
        user_answer = render_answer_widget(row, options, f"learn_{idx}")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        if st.button("Antwort prüfen", use_container_width=True):
            st.session_state.learn_checked = True
    with c2:
        if st.button("Vorherige Frage", use_container_width=True, disabled=ptr == 0):
            st.session_state.learn_checked = False
            st.session_state.learning_pointer = max(0, ptr - 1)
            st.rerun()
    with c3:
        if st.button("Nächste Frage", use_container_width=True, disabled=ptr >= len(indices) - 1):
            st.session_state.learn_checked = False
            st.session_state.learning_pointer = min(len(indices) - 1, ptr + 1)
            st.rerun()
    with c4:
        if st.button("Lernmodus zurücksetzen", use_container_width=True):
            reset_learning()
            st.rerun()

    if st.session_state.learn_checked:
        if check_answer(user_answer, correct, row["question_type"]):
            st.success("Richtige Antwort ✅")
        else:
            st.error("Leider falsch ❌")
        st.info(f"Korrekte Antwort: {format_answer_set(correct, options)}")
        if str(row.get("explanation", "")).strip():
            st.caption(f"Erklärung: {row.get('explanation', '')}")


def render_exam_mode(df: pd.DataFrame, random_order: bool) -> None:
    st.header("Prüfungsmodus")
    if df.empty:
        st.info("Keine Fragen nach aktuellem Filter verfügbar.")
        return

    n_questions = st.sidebar.number_input("Anzahl Prüfungsfragen", 1, len(df), min(10, len(df)), 1)

    if not st.session_state.exam_started:
        st.info("In der Prüfung werden keine Lösungen angezeigt. Antworten werden nur gespeichert.")
        if st.button("Prüfung starten", type="primary"):
            st.session_state.exam_started = True
            st.session_state.exam_finished = False
            st.session_state.exam_pointer = 0
            st.session_state.exam_answers = {}

            pool = list(df.index)
            if random_order:
                random.shuffle(pool)
            st.session_state.exam_indices = pool[: int(n_questions)]
            st.rerun()
        return

    if st.session_state.exam_finished:
        show_exam_results(df)
        if st.button("Prüfung zurücksetzen"):
            reset_exam()
            st.rerun()
        return

    ptr = st.session_state.exam_pointer
    exam_indices = st.session_state.exam_indices
    if not exam_indices:
        st.warning("Es sind keine Prüfungsfragen ausgewählt. Bitte Prüfung neu starten.")
        reset_exam()
        return
    idx = exam_indices[ptr]
    row = df.loc[idx]
    options = get_answer_options(row)

    st.progress((ptr + 1) / len(exam_indices), text=f"Prüfungsfrage {ptr + 1} von {len(exam_indices)}")
    with st.container(border=True):
        st.subheader(row["question_text"])
        user_answer = render_answer_widget(row, options, f"exam_{idx}")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Antwort speichern", use_container_width=True):
            st.session_state.exam_answers[idx] = sorted(user_answer)
            st.success("Antwort gespeichert.")
    with c2:
        next_label = "Nächste Frage" if ptr < len(exam_indices) - 1 else "Prüfung beenden"
        if st.button(next_label, use_container_width=True):
            st.session_state.exam_answers[idx] = sorted(user_answer)
            if ptr < len(exam_indices) - 1:
                st.session_state.exam_pointer += 1
            else:
                st.session_state.exam_finished = True
            st.rerun()
    with c3:
        if st.button("Prüfung abbrechen", use_container_width=True):
            reset_exam()
            st.rerun()


def show_exam_results(df: pd.DataFrame) -> None:
    st.subheader("Ergebnis")
    if not st.session_state.exam_indices:
        st.info("Keine Ergebnisse vorhanden. Bitte zuerst eine Prüfung starten.")
        return

    points = 0
    rows = []

    for idx in st.session_state.exam_indices:
        row = df.loc[idx]
        options = get_answer_options(row)
        correct = parse_correct_answers(row["correct_answers"], row["question_type"], options)
        user = set(st.session_state.exam_answers.get(idx, []))
        is_ok = check_answer(user, correct, row["question_type"])
        points += int(is_ok)

        rows.append(
            {
                "Frage": row["question_text"],
                "Eigene Antwort": format_answer_set(user, options),
                "Korrekte Antwort": format_answer_set(correct, options),
                "Erklärung": row.get("explanation", ""),
                "Status": "Richtig" if is_ok else "Falsch",
            }
        )

    total = len(st.session_state.exam_indices)
    percent = (points / total * 100) if total else 0.0
    passed = percent >= st.session_state.pass_threshold

    c1, c2, c3 = st.columns(3)
    c1.metric("Punkte", f"{points}/{total}")
    c2.metric("Prozent", f"{percent:.1f}%")
    c3.metric("Bestehensgrenze", f"{st.session_state.pass_threshold}%")

    if passed:
        st.success("Bestanden ✅")
    else:
        st.error("Nicht bestanden ❌")

    st.dataframe(pd.DataFrame(rows), use_container_width=True)




def main() -> None:
    st.set_page_config(page_title="Prüfungstrainer", page_icon="📖", layout="wide")
    init_state()

    st.markdown(
        """
<style>
.block-container {padding-top: 1.4rem;}
h1, h2, h3 {letter-spacing: 0.2px;}
</style>
""",
        unsafe_allow_html=True,
    )

    st.title("Prüfungstrainer")

    with st.sidebar:
        st.header("Einstellungen")

    try:
        raw_df = load_questions()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"CSV konnte nicht geladen werden: {e}")
        st.stop()

    ok, error = validate_columns(raw_df)
    if not ok:
        st.error(error)
        st.stop()

    questions_df, warnings = sanitize_questions(raw_df)
    for msg in warnings:
        st.warning(msg)

    if questions_df.empty:
        st.error("Keine gültigen Fragen in der CSV gefunden.")
        st.stop()

    render_header_stats(questions_df)

    topics = ["Alle"] + sorted([t for t in questions_df["topic"].unique() if str(t).strip()])
    difficulties = ["Alle"] + sorted([d for d in questions_df["difficulty"].unique() if str(d).strip()])
    question_types = ["Alle"] + sorted(questions_df["question_type"].unique())

    with st.sidebar:
        mode = st.radio("Modus", ["Lernmodus", "Prüfungsmodus"])
        topic = st.selectbox("Thema", topics)
        difficulty = st.selectbox("Schwierigkeit", difficulties)
        qtype = st.selectbox(
            "Fragetyp",
            question_types,
            format_func=lambda x: "Alle" if x == "Alle" else QUESTION_TYPE_LABELS.get(x, x),
        )
        random_order = st.toggle("Zufällige Reihenfolge", value=True)
        st.session_state.pass_threshold = 60

    signature = f"{mode}|{topic}|{difficulty}|{qtype}|{random_order}"
    maybe_reset_on_context_change(signature)

    filtered = apply_filters(questions_df, topic, difficulty, qtype)
    if filtered.empty:
        st.warning("Kein Treffer für die gewählten Filter. Bitte Filter anpassen.")
        return

    if mode == "Lernmodus":
        render_learning_mode(filtered, random_order)
    else:
        render_exam_mode(filtered, random_order)


if __name__ == "__main__":
    main()
