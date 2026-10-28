# Pitch Predictor

A full-stack machine learning app that estimates Dylan Cease's next pitch type from the current game situation. A React and TypeScript frontend sends inputs to a FastAPI backend, which returns probabilities from a saved random forest model.

## Features

- Enter the count, batter handedness, inning, outs, and previous pitch.
- View estimated probabilities for each pitch type as horizontal bars.
- Automatically use `START` for the previous pitch at a 0–0 count.
- Use the predictor through the web app, API, or interactive terminal script.
- Download historical pitch data and reproduce data preparation, training, and evaluation.

## Tech stack

| Area | Tools |
| --- | --- |
| Data | Python, pandas, pybaseball, Baseball Savant / Statcast |
| Machine learning | scikit-learn, random forest |
| Model persistence | joblib |
| Backend | FastAPI, Uvicorn |
| Frontend | React, TypeScript, Vite, CSS |
| Code checks | TypeScript compiler, ESLint |

## Model and data

The first version uses Dylan Cease's 2025 regular-season data: 3,052 records across 32 games. Removing 8 records without pitch-type labels leaves 3,044 labeled pitches.

The model uses six inputs:

| Input | Description |
| --- | --- |
| `balls` | Balls before the pitch, from 0 to 3 |
| `strikes` | Strikes before the pitch, from 0 to 2 |
| `stand` | Batter's side of the plate: `L` or `R` |
| `inning` | Current inning |
| `outs_when_up` | Outs before the pitch, from 0 to 2 |
| `previous_pitch` | Previous pitch type in the same plate appearance, `START`, or `UNKNOWN` |

Previous-pitch features are created in chronological order within each plate appearance, before removing unlabeled records. Current-pitch measurements such as speed, spin, and location are excluded because they would not be available before the pitch.

The random forest uses 200 trees, maximum depth 6, minimum leaf size 20, and random seed 42. Categorical inputs are encoded inside the saved preprocessing pipeline.

## Evaluation

Games are split chronologically and kept whole. Initial development used 19 games for training and the following 6 for validation. Additional expanding-window checks trained on games 1–10, 1–15, and 1–20, validating on the following five games each time.

Across those three development windows, the random forest achieved 51.1% combined accuracy versus 47.0% for a smoothed count-and-handedness baseline. These are development results, not an independent final test.

For the final evaluation, the model trained on the first 25 games (2,360 pitches) and was evaluated on the last 7 games (684 pitches), dated August 21 through September 24, 2025.

| Predictor | Final test accuracy ↑ | Final test log loss ↓ |
| --- | ---: | ---: |
| Training-frequency baseline | 31.6% | 1.476 |
| Smoothed count-and-handedness baseline | 35.5% | 1.434 |
| Random forest | **40.1%** | **1.398** |

The forest improves accuracy by 4.6 percentage points over the situational baseline. Lower log loss indicates better probability predictions on this test set; it does not by itself establish probability calibration.

These figures document the original evaluation. Download revisions or different dependency versions may produce slightly different results. The final test games have now been inspected; improvements informed by them require fresh held-out data for another independent evaluation.

## Project structure

```text
pitch-predictor/
├── data/                       # Generated datasets; ignored by Git
├── models/                     # Saved model files; .joblib files ignored
├── src/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── validate_model.py
│   ├── test_model.py
│   ├── pitch_predictor.py       # Reusable prediction function
│   ├── predict.py               # Interactive terminal interface
│   └── api.py                   # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   └── index.css
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.ts
├── requirements.txt
├── .gitignore
└── README.md
```

## Local setup

Requires Python 3.10 or later, Node.js compatible with the project's Vite version, and npm. Internet access is needed to install dependencies and download pitch data.

Clone this repository, open its root folder in a terminal, and create a Python environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Use `python3` instead of `python` when needed to create the environment. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

### Generate the dataset and saved model

Run these commands from the repository root, in order:

```bash
python src/download_data.py
python src/prepare_data.py
python src/test_model.py
```

The last command trains on the first 25 games, reports results on the last 7, and saves `models/dylan_cease_v1.joblib`. Data and model files must be generated on a fresh clone because they are excluded from Git. Repeating the evaluation does not make it a new independent test.

Optional inspection and development evaluation:

```bash
python src/inspect_data.py
python src/train_model.py
python src/validate_model.py
```

### Start the backend

From the repository root with the Python environment active:

```bash
python -m uvicorn api:app --app-dir src --reload
```

- Backend: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

### Start the frontend

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite, usually http://localhost:5173. Keep both servers running while using the app.

The Vite development proxy forwards `/api/predict` to the backend's `/predict` endpoint. Production hosting will need its own API routing configuration.

### Terminal predictor

From the repository root with the Python environment active:

```bash
python src/predict.py
```

Follow the prompts to enter a situation and receive pitch probabilities.

## API example

Send a `POST` request to `/predict` with this JSON body:

```json
{
  "balls": 1,
  "strikes": 2,
  "stand": "L",
  "inning": 6,
  "outs_when_up": 1,
  "previous_pitch": "FF"
}
```

The response contains `pitcher` and a `probabilities` array. Each array item contains `pitch_type` and `probability`, sorted from highest to lowest probability. Probabilities are decimals: `0.429` represents 42.9%.

Pitch codes: `FF` = four-seam fastball, `SL` = slider, `KC` = knuckle curve, `CH` = changeup, `SI` = sinker, and `ST` = sweeper.

## Frontend checks

From `frontend/`:

```bash
npm run build
npm run lint
```

Manual checks include submitting a prediction, checking the 0–0 previous-pitch behavior, and confirming that a failed backend request displays an error.

## Limitations

- The model covers one pitcher and one season. It does not currently support selecting a specific batter or another pitcher.
- In the original final test, only fastball and slider were selected as top predictions. No other pitch types were correctly identified, although they received nonzero probabilities.
- Rare situations can have very few examples. Overall accuracy does not guarantee reliable estimates at every count or matchup.
- Probabilities have not been calibrated. A displayed 60% estimate is not a verified 60% real-world frequency.
- The interfaces assume a 0–0 count starts a plate appearance. Unusual count changes and automatic ball/strike events are not fully modeled.
- Inputs are entered manually; live game data is not connected.

## Possible next steps

- Add more seasons and evaluate on fresh games.
- Compare models and inspect performance by count and pitch type.
- Add more pitchers and batter-specific information.
- Improve reproducibility through dependency pinning and dataset versioning.
- Deploy the frontend and backend.

## Data source

Historical pitch data comes from [Baseball Savant's Statcast database](https://baseballsavant.mlb.com/statcast_search), retrieved using [pybaseball](https://github.com/jldbc/pybaseball). Field definitions are available in the [Statcast CSV documentation](https://baseballsavant.mlb.com/csv-docs).
