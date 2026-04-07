# AI-Powered Rehabilitation Coaching System — Local Full Streamlit App

-   continuous browser video via `streamlit-webrtc`
-   local checkpoint inference with MediaPipe overlay
-   the integration layer for temporal filtering, deduplication, routing, and Tier 1 cache
-   Anthropic Claude Sonnet for live Tier 2 / Tier 3 coaching
-   Ollama `gemma3:4b` for the end-of-session text report
-   optional text-to-speech, limited to Tier 2 / Tier 3 and rate-limited to one spoken cue every 5 seconds
-   a patient context text box whose content is injected into live coaching and report generation, then cleared on reset

## What is included

-   `streamlit_app.py` — main app
-   `app_wrapper.py` — wrapper around the repo integration and report components
-   `live_infer_stream_engine.py` — streamlit frame-by-frame inference engine adapted from `infer_stream_v2.py`
-   `speech_manager.py` — TTS queue, tier gating, and 5-second spacing
-   `infer_stream_v2.py` — your local inference script reference, used as the model/pose runtime base
-   `requirements_full_local_mvp.txt`

## Recommended environment on Apple Silicon (M4 Max, macOS 15.7.4)

PyTorch supports the `mps` device on Apple silicon, and Apple documents MPS acceleration for PyTorch on Mac. Use Python 3.11 in a fresh conda env, then install current pip packages on top.

``` bash
conda create -n rehab_streamlit_full python=3.11 -y
conda activate rehab_streamlit_full
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_full_local_mvp.txt
```

### Notes for Apple Silicon

-   Leave the app device on `auto` first. It will prefer `mps` when available.
-   If a Torch op fails on MPS for your checkpoint, switch the sidebar device to `cpu`.

## Required local assets

You need these on your Mac before running the app:

1.  **Your checkpoint file**
2.  **The rehab corpus** at `dataset/clean/` for report retrieval
3.  **Download entire github repo and place in same directory as app**
4.  **Anthropic API key** for live Tier 2 / Tier 3 coaching
5.  **Ollama** installed locally for reports

Install Ollama and pull the report model:

``` bash
curl -fsSL https://ollama.com/install.sh | sh      
ollama serve
ollama pull gemma3:4b
```

## Run

``` bash
export ANTHROPIC_API_KEY= ####
streamlit run streamlit_app.py
```

The script: - loads `.env` if present - prompts you for `ANTHROPIC_API_KEY` if it is not already exported - starts `ollama serve` if needed - pulls `gemma3:4b` if needed - launches Streamlit

## How the app maps to the repo

-   Live coaching uses the repo integration graph, which routes Tier 2 and Tier 3 through Anthropic-backed generation and polishing.
-   The repo environment description includes OpenAI, Anthropic, and Ollama clients along with LangChain, LangGraph, ChromaDB, and sentence-transformers.
-   Session reports use the repo’s Ollama/Chroma progress-tracker stack.
-   TTS is built on the repo’s `src/text_to_voice/tts.py`, but wrapped so it only speaks Tier 2 / Tier 3 cues and enforces a 5-second spacing rule.

## Usage notes

-   Use the **camera component’s own START/STOP controls** to open and close the webcam stream.
-   Use **Start Coaching / Stop Coaching** in the app to enable or disable feedback generation.
-   Use **Generate Report** after a session to run the Ollama report.
-   Use **Reset Session** to clear transcript state and delete the session-only patient context note.
-   TTS is off by default and can be enabled in the sidebar.

## Troubleshooting

### Webcam stream does not start

-   Grant browser camera permissions
-   Try Chrome first if Safari blocks WebRTC
-   Make sure localhost networking is allowed

### Live coaching fails but Tier 1 still works

-   Check that `ANTHROPIC_API_KEY` is valid
-   Confirm your network allows Anthropic API calls

### Report generation fails

-   Confirm `ollama serve` is running
-   Confirm `ollama list` shows `gemma3:4b`
-   Confirm `dataset/clean/` contains `.txt` and/or `.html` rehab documents
