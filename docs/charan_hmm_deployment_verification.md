# Charan's contribution: HMM decoding and distributed deployment

This bundle contains only the assigned HMM, map-validation and deployment-verification components.

## Files
- `pathsnap_real_hmm.py`: fixed directed HMM/Viterbi decoder and development-only evaluation.
- `pathsnap_real_map.py`: GPS, classifier, HMM and reference route overlay for permitted student-test trips.
- `pathsnap_real_api.py` + `pathsnap_real_ui.html`: student-testing API and dashboard.
- `pathsnap_evaluator_api.py` + `pathsnap_evaluator_ui.html`: separately hosted reserved-evaluator API/UI. Do not use reserved trips for development, tuning or model selection.
- `pathsnap_combined_ui.html`: dashboard switcher for student API port 8002 and evaluator API port 8003.
- `artifacts/official_tuning/`: frozen tuned model and its feature configuration.
- `results/official_hmm/decoder_config.json`: fixed decoder parameters.

## Fixed decoder configuration
- `beta = 0.5`
- `max_hops = 4`
- `disconnected_penalty = 12.0`
- `gap_seconds = 60.0`
- Parameter selection is explicitly marked as initial fixed configuration, not optimized.

## Install and run (Laptop 1: API server)
From the repository root, create/activate a virtual environment and install `requirements.txt`.
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m uvicorn pathsnap_real_api:app --host 0.0.0.0 --port 8002
```
Only the `data/student_test` partition is served by this API.

When reserved evaluation is explicitly authorised, start the separate evaluator API in another terminal:
```powershell
py -m uvicorn pathsnap_evaluator_api:app --host 0.0.0.0 --port 8003
```
The evaluator API must remain separate; do not copy reserved data into student-test folders or use it for decoder/model changes.

## Laptop 2: dashboard
Copy `pathsnap_combined_ui.html` to a folder on Laptop 2 and run:
```powershell
py -m http.server 8083 --bind 0.0.0.0
```
Open `http://localhost:8083/pathsnap_combined_ui.html`. Set the server address to `http://<LAPTOP_1_IP>:8002` for student testing. Select the reserved evaluator mode only during authorised evaluation; the dashboard switches to port 8003.

Both laptops should use the same trusted network. Allow inbound TCP ports 8002 and 8083 (and 8003 only when authorised) through the host firewall. Keep the APIs on a trusted network; they are not authenticated public services. Use the current Laptop 1 LAN IP, not `localhost`, on Laptop 2.

## Verification
Run decoder tests:
```powershell
py -m pytest tests/test_pathsnap_hmm.py tests/test_api_smoke.py -q\n# Live API checks (after starting student API):\n$env:PATHSNAP_API_URL = "http://127.0.0.1:8002"\npy -m pytest tests/test_api_smoke.py -q
```
Student API smoke checks (with Laptop 1 API running):
```powershell
Invoke-RestMethod http://127.0.0.1:8002/
Invoke-RestMethod http://127.0.0.1:8002/trips
```
Then request one returned student trip ID:
```powershell
Invoke-RestMethod http://127.0.0.1:8002/predict/<TRIP_ID>
```
Open `http://127.0.0.1:8002/docs` for the API schema. The `/map/<TRIP_ID>` endpoint returns the interactive map.

## Data requirement and limitation
The new `ML-HACKATHON-126` checkout previously showed no `data/` files. The API, map generator and end-to-end inference therefore cannot be fully exercised until the organiser-provided permitted development data is placed locally in the expected `data/student_test/<trip_id>/` layout. Do not add datasets to Git. The included tests exercise the decoder without requiring any trip data. The model artifact is frozen and included so deployment does not require retraining.\n