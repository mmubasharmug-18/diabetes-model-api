# Diabetes Prediction API with Audit Logging

A FastAPI service that serves a trained diabetes model. You send 8 patient measurements to `/predict` and get back a prediction and the probabilities. Every `/predict` request, successful or not, is written to a SQLite database so there is a record of what was asked and what happened.

The model is a Random Forest trained on the Pima Indians Diabetes dataset (768 rows).

## Setup (VS Code)

1. Open the `diabetes_model_api` folder in VS Code.
2. Make and activate a virtual environment in the terminal:

   ```
   python -m venv .venv
   ```

   - Windows: `.venv\Scripts\activate`
   - Mac or Linux: `source .venv/bin/activate`

3. Install the packages:

   ```
   pip install -r requirements.txt
   ```

4. Download `diabetes.csv` from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database and put it in the `data` folder.
5. Train the model:

   ```
   python -m ml.train
   ```

   This saves `models/model.joblib` and `models/model_info.json` (version, row count, test scores) and prints the scores.
6. Start the API:

   ```
   uvicorn app.main:app --reload
   ```

Swagger docs are at http://127.0.0.1:8000/docs. If you start the API before training, it still runs, but `/predict` answers 503 until the model file exists.

## Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| POST | `/predict` | Validates the input, returns the prediction and probabilities, and logs the request |
| GET | `/health` | Says if the service is up and the model is loaded |
| GET | `/audit` | Recent logged requests, newest first. Optional `?limit=` (1 to 200) and `?status=success` or `failure` |
| GET | `/audit/summary` | Counts of successful and failed requests |

## Example requests

The request bodies are in the `samples` folder, so the commands are the same on every system. On Windows PowerShell use `curl.exe` instead of `curl`.

```
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/valid.json

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/valid_2.json
```

A successful answer looks like this (the numbers come from your model):

```json
{
  "request_id": "9f0c5b1e6d2a4c1f8e3b7a5d4c2e1f60",
  "prediction": 1,
  "label": "diabetes",
  "probability": { "no_diabetes": 0.12, "diabetes": 0.88 },
  "model_version": "rf-20261023-073720"
}
```

Bad input gets a 400 that says what is wrong. Add `-i` to see the status line.

```
curl -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/invalid_range.json

curl -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/invalid_types.json
```

```json
{
  "detail": [
    { "field": "glucose", "message": "glucose must be a number between 40 and 400 (got 520)" },
    { "field": "age", "message": "age must be a whole number between 21 and 120 (got 15)" }
  ],
  "request_id": "3b8e0d7a5c1f4e2a9d6b0c8f7e1a2d45"
}
```

Postman: make a **POST** request to `http://127.0.0.1:8000/predict`, open Body, choose **raw** and **JSON**, paste the contents of `samples/valid.json` and press Send. Change `glucose` to `520` and send again to see the 400. The VS Code REST Client extension can run everything in `requests.http` too.

## Input rules

All eight fields are required. Text in place of a number, `true`/`false` and `null` are rejected.

| Field | Type | Allowed range |
| --- | --- | --- |
| pregnancies | whole number | 0 to 20 |
| glucose | number (mg/dL) | 40 to 400 |
| blood_pressure | number (diastolic, mm Hg) | 20 to 200 |
| skin_thickness | number (mm) | 0 to 100 |
| insulin | number (mu U/ml) | 0 to 900 |
| bmi | number | 10 to 70 |
| diabetes_pedigree_function | number | 0 to 3 |
| age | whole number | 21 to 120 |

For skin thickness and insulin, `0` means "not measured". The dataset uses 0 that way for several columns, so the model treats a 0 in glucose, blood pressure, skin thickness, insulin or BMI as a missing value and fills it with the median from training.

## Audit trail

**What is logged.** Every `POST /predict`:

- **success** (200): the input, the prediction, the probability of diabetes, the time taken and the model version.
- **failure, 400**: input that did not pass validation, including broken JSON. The input as received and the exact error messages are stored.
- **failure, 503**: the model file was not loaded.
- **failure, 500**: anything unexpected while predicting. The error type and message are stored.

Not logged: `/health`, `/audit`, `/docs` and unknown URLs. If writing to the database itself fails, the prediction is still returned and the problem goes to the server log.

**Where.** `logs/audit.db`, table `request_log`:

| Column | Meaning |
| --- | --- |
| id | row number |
| request_id | same id the caller got in the response |
| timestamp | UTC time |
| endpoint | always `/predict` for now |
| status | `success` or `failure` |
| http_status | 200, 400, 503 or 500 |
| input_json | the input as received |
| prediction, probability | result (success only) |
| error | reason (failure only) |
| latency_ms | time spent on the prediction |
| model_version | which model answered |

**Looking at it.** Three ways:

- API: http://127.0.0.1:8000/audit and http://127.0.0.1:8000/audit/summary
- Straight from the file: `python scripts/show_audit.py` (add `--status failure` to see only failures)
- In VS Code: install the "SQLite Viewer" extension and open `logs/audit.db`

**Demo.** With the server running, open a second terminal and run:

```
python scripts/demo_audit_trail.py
```

It sends three valid requests, then three bad ones (out of range, wrong types, broken JSON), and prints the audit trail. This is an example of the output from a test run. Your probabilities and times will be different because they come from your own model:

```
valid patient 1                          -> HTTP 200 (diabetes)
valid patient 2                          -> HTTP 200 (no diabetes)
valid patient 3                          -> HTTP 200 (diabetes)
glucose and age out of range             -> HTTP 400 (rejected)
wrong data types and a missing field     -> HTTP 400 (rejected)
broken JSON                              -> HTTP 400 (rejected)

 id  time (UTC)  status   http  pred    prob      ms  error
-----------------------------------------------------------
  1  07:37:29    success   200     1   0.877    54.6
  2  07:37:29    success   200     0   0.075    41.7
  3  07:37:29    success   200     1   0.661    45.1
  4  07:37:29    failure   400                        glucose must be a number between 40 and 400 (got 520); ag...
  5  07:37:29    failure   400                        pregnancies must be a whole number between 0 and 20 (got ...
  6  07:37:29    failure   400                        request body must be valid JSON

total 6 | successes 3 | failures 3 | success rate 50% | failures by status {'400': 3}
```

To trace one request, take the `request_id` from a response and find the same id in the log. Every request, good or bad, gets one.

## Tests

```
pytest
```

- `tests/unit`: input validation, error messages, the audit log, the model service and training (these use a small stand-in dataset, so you do not need the Kaggle file).
- `tests/integration`: the real endpoints through FastAPI's test client. Valid requests, every range limit, wrong types, missing fields, broken JSON, 503 and 500 cases, and checks that each kind of request ends up in the audit log.

## Folder structure

```
diabetes_model_api/
  app/
    main.py            creates the app and loads the model on start
    config.py          file paths
    schemas.py         input rules and response models
    errors.py          readable validation messages
    handlers.py        400 responses, and logging of rejected requests
    model_service.py   loads the model and makes predictions
    audit_log.py       SQLite audit log
    state.py           the one model service and audit log the app shares
    routes/
      predict.py       POST /predict
      health.py        GET /health
      audit.py         GET /audit and /audit/summary
  ml/
    features.py        column names and the zero-means-missing step
    train.py           trains and saves the model
  data/                put diabetes.csv here
  models/              the trained model is saved here
  logs/                audit.db is created here
  scripts/             demo_audit_trail.py and show_audit.py
  samples/             request bodies for curl and Postman
  tests/
  requirements.txt
  README.md
```

## Limitations

- **Not a medical tool.** The model learned from 768 rows from one study (adult women of Pima heritage). It will not generalise to everyone, and it is only a learning exercise.
- **Modest accuracy.** The test scores in `models/model_info.json` come from a single 80/20 split with no tuning. With a test set of about 150 people, a few cases change the scores a lot.
- **Fixed threshold.** The label is "diabetes" when its probability is higher than the other class (50%). A lower cut-off would catch more cases but give more false alarms.
- **Input is stored.** The audit log keeps the full input of each request. Use made-up data only, not real patient information.
- **No login.** Anyone who can reach the server can predict and read `/audit`.
- **SQLite is for light use.** It is one file with one writer at a time, which is fine here but not for heavy traffic.
- **Model changes need a restart.** Retrain, then restart the server so the new model is loaded. The model file must be loaded with the same scikit-learn version that trained it.
- **Only `/predict` is audited.** Other URLs are not logged.
