# Streamlit Prüfungstrainer

Eine einfache Prüfungstrainer-App auf Basis von **Streamlit** und **CSV-Fragen**.

## Zweck
- Lernen und Prüfen mit Single Choice, Multiple Choice und Richtig/Falsch.
- Fragen werden lokal aus einer CSV-Datei geladen.
- Optional kann eine eigene CSV über die Sidebar hochgeladen werden.

## CSV-Struktur
Die App erwartet folgende Spalten:

```csv
question_text,question_type,topic,difficulty,explanation,option_a,option_b,option_c,option_d,correct_answers
```

Beispielwerte für `question_type`:
- `single_choice` / `single choice`
- `multiple_choice` / `multiple choice`
- `true_false` / `richtig/falsch` / `wahr/falsch`

## Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hinweis
Diese App benötigt **keine Datenbank**, **kein Backend**, **kein Next.js**, **kein Prisma** und **keine Cloud-Abhängigkeiten**.
