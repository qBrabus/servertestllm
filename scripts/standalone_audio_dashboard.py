"""
Dashboard FastAPI autonome pour piloter les APIs d'inférence existantes.

Fonctionnalités principales :
- Afficher l'état GPU / système via `/api/admin/status`.
- Uploader un fichier audio (français/anglais/allemand) et le transcrire via Canary.
- Réaliser la diarisation Pyannote puis renommer les locuteurs avec des rôles contextuels.
- Traduire la transcription en français, allemand et anglais.
- Exposer une page HTML minimaliste (sans dépendance frontend) pour visualiser les résultats.

Ce service est volontairement autonome : il s'exécute hors du conteneur principal
et interroge l'API existante pointée par `UPSTREAM_API_BASE`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Mapping, MutableMapping

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

API_BASE = os.getenv("UPSTREAM_API_BASE", "http://10.200.50.45:8000")
API_TOKEN = os.getenv("UPSTREAM_API_TOKEN")
CHAT_MODEL = os.getenv("UPSTREAM_CHAT_MODEL", "qwen")
REQUEST_TIMEOUT = httpx.Timeout(180.0)

app = FastAPI(title="Standalone audio dashboard", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Utilitaires HTTP ----------

def _auth_headers() -> Dict[str, str]:
    if not API_TOKEN:
        return {}
    return {"Authorization": f"Bearer {API_TOKEN}"}


def _clean_json_payload(text: str) -> Mapping[str, object]:
    """Tente de parser le contenu JSON renvoyé par le modèle (code fences tolérés)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json\n"):
            cleaned = cleaned[5:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:  # pragma: no cover - protection runtime
        raise HTTPException(status_code=502, detail="Impossible de parser la sortie JSON du modèle") from exc


# ---------- Préparation des segments ----------

def _fallback_segments(transcription: Mapping[str, object]) -> List[Mapping[str, object]]:
    segments = transcription.get("segments")
    if isinstance(segments, list) and segments:
        return segments  # type: ignore[return-value]
    # Pas de segments détaillés : on retourne un unique bloc couvrant tout le texte.
    return [
        {
            "text": transcription.get("text", ""),
            "start": 0.0,
            "end": 0.0,
        }
    ]


def _collect_text_in_window(
    segments: Iterable[Mapping[str, object]], start: float, end: float
) -> str:
    collected: List[str] = []
    for seg in segments:
        seg_start = seg.get("start")
        seg_end = seg.get("end")
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if seg_start is None or seg_end is None:
            collected.append(text)
            continue
        try:
            seg_start_f = float(seg_start)
            seg_end_f = float(seg_end)
        except (TypeError, ValueError):
            collected.append(text)
            continue
        overlap = not (seg_end_f < start or seg_start_f > end)
        if overlap:
            collected.append(text)
    return " ".join(collected)


def _aggregate_by_speaker(
    diar_segments: Iterable[Mapping[str, object]], trans_segments: List[Mapping[str, object]]
) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = {}
    for segment in diar_segments:
        speaker = str(segment.get("speaker", "speaker"))
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        text = _collect_text_in_window(trans_segments, start, end)
        if text:
            grouped.setdefault(speaker, []).append(text)
    return {spk: " ".join(parts) for spk, parts in grouped.items()}


def _decorate_diarization(
    diar_segments: List[MutableMapping[str, object]], labels: Mapping[str, str], trans_segments: List[Mapping[str, object]]
) -> List[Mapping[str, object]]:
    enriched: List[Mapping[str, object]] = []
    for seg in diar_segments:
        speaker = str(seg.get("speaker", "speaker"))
        role = labels.get(speaker, speaker)
        enriched.append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "speaker": speaker,
                "role": role,
                "text": _collect_text_in_window(trans_segments, float(seg.get("start", 0.0)), float(seg.get("end", 0.0))),
            }
        )
    return enriched


# ---------- Requêtes modèle ----------

async def _run_completion(client: httpx.AsyncClient, messages: List[Mapping[str, str]]) -> Mapping[str, object]:
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.2}
    resp = await client.post("/v1/chat/completions", json=payload, headers=_auth_headers())
    resp.raise_for_status()
    data = resp.json()
    try:
        message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - protection runtime
        raise HTTPException(status_code=502, detail="Réponse modèle invalide") from exc
    return _clean_json_payload(message)


async def _translate_transcript(client: httpx.AsyncClient, english_text: str) -> Mapping[str, str]:
    prompt = (
        "Tu es un traducteur professionnel. Produis uniquement un JSON avec les clés "
        "'english', 'french', 'german'. La valeur 'english' doit être une version "
        "nettoyée et lisible du texte d'entrée. Traduis fidèlement et sans commentaire."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": english_text},
    ]
    data = await _run_completion(client, messages)
    return {
        "english": str(data.get("english", english_text)),
        "french": str(data.get("french", "")),
        "german": str(data.get("german", "")),
    }


async def _label_speakers(
    client: httpx.AsyncClient, speaker_texts: Mapping[str, str], context: str
) -> Mapping[str, str]:
    descriptions = [f"- {speaker}: {text[:300]}" for speaker, text in speaker_texts.items()]
    guide = (
        "Attribue un rôle descriptif en français (journaliste, médecin, patient, président, expert, témoin, etc.) "
        "pour chaque locuteur, en t'appuyant sur le contexte global. Réponds au format JSON {""speaker"": ""rôle""}. "
        "Conserve les noms de locuteur tels qu'ils apparaissent et choisis un rôle court (1 à 3 mots)."
    )
    messages = [
        {"role": "system", "content": guide},
        {
            "role": "user",
            "content": f"Transcription anglaise :\n{context}\n\nLocuteurs :\n" + "\n".join(descriptions),
        },
    ]
    return await _run_completion(client, messages)


# ---------- Routes API ----------

@app.get("/", response_class=HTMLResponse)
async def serve_index() -> str:
    return INDEX_HTML


@app.get("/status")
async def proxy_status() -> JSONResponse:
    async with httpx.AsyncClient(base_url=API_BASE, timeout=REQUEST_TIMEOUT) as client:
        resp = await client.get("/api/admin/status", headers=_auth_headers())
    resp.raise_for_status()
    return JSONResponse(resp.json())


@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Fichier audio vide")

    async with httpx.AsyncClient(base_url=API_BASE, timeout=REQUEST_TIMEOUT) as client:
        # Transcription Canary
        trans_resp = await client.post(
            "/api/audio/transcribe",
            files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            headers=_auth_headers(),
        )
        trans_resp.raise_for_status()
        transcription = trans_resp.json()
        trans_segments = _fallback_segments(transcription)

        # Diarisation Pyannote
        diar_resp = await client.post(
            "/api/diarization/process",
            files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            headers=_auth_headers(),
        )
        diar_resp.raise_for_status()
        diarization = diar_resp.json().get("segments", [])

        english_text = transcription.get("text", "")
        translations = await _translate_transcript(client, english_text)

        if diarization:
            speaker_texts = _aggregate_by_speaker(diarization, trans_segments)
            labels = await _label_speakers(client, speaker_texts, translations["english"])
            timeline = _decorate_diarization(diarization, labels, trans_segments)
        else:
            labels = {}
            timeline = []

    payload = {
        "translations": translations,
        "timeline": timeline,
        "speaker_labels": labels,
        "raw": {
            "transcription": transcription,
            "diarization": diarization,
        },
    }
    return JSONResponse(payload)


# ---------- Gabarit HTML ----------

INDEX_HTML_TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Audio dashboard autonome</title>
  <style>
    :root { font-family: 'Inter', system-ui, sans-serif; background: #0b1726; color: #e7edf5; }
    body { max-width: 1200px; margin: 0 auto; padding: 24px; }
    h1 { margin-bottom: 8px; }
    section { background: #111b2d; border: 1px solid #1f2a40; border-radius: 10px; padding: 16px 18px; margin-bottom: 18px; }
    button { background: #198fff; border: none; color: white; padding: 10px 14px; border-radius: 8px; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
    .card { background: #0f2238; border: 1px solid #1f3653; border-radius: 8px; padding: 12px; }
    .badge { display: inline-block; padding: 2px 6px; background: #0b5dab; border-radius: 6px; font-size: 12px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px; text-align: left; border-bottom: 1px solid #1f2a40; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0f1826; padding: 12px; border-radius: 8px; border: 1px solid #1f2a40; }
  </style>
</head>
<body>
  <h1>Audio dashboard autonome</h1>
  <p>Statut GPU en direct et pipeline Canary + Pyannote (transcription → diarisation → traduction).</p>

  <section>
    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
      <div>
        <div class="badge">Backend</div>
        <div>UPSTREAM_API_BASE = <code id="api-base"></code></div>
      </div>
      <button id="refresh-status" onclick="refreshStatus()">Actualiser le statut</button>
    </div>
    <div id="status" style="margin-top:12px;" class="grid"></div>
  </section>

  <section>
    <h2>Uploader un audio</h2>
    <p>Langues acceptées : français, anglais, allemand. La transcription est normalisée en anglais, puis traduite en trois langues. Les rôles des locuteurs sont inférés contextuellement (journaliste, médecin, patient, président...).</p>
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
      <input type="file" id="file-input" accept="audio/*" />
      <button id="submit-audio" onclick="uploadAudio()">Lancer le traitement</button>
      <span id="upload-status"></span>
    </div>
  </section>

  <section>
    <h2>Résultats</h2>
    <div id="translations"></div>
    <div id="timeline"></div>
    <h3>Brut (debug)</h3>
    <pre id="raw"></pre>
  </section>

<script>
  const apiBase = "" + "__API_BASE__";
  document.getElementById('api-base').innerText = apiBase;

  async function refreshStatus() {
    const container = document.getElementById('status');
    container.innerHTML = 'Chargement...';
    try {
      const res = await fetch('/status');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      renderStatus(data);
    } catch (err) {
      container.innerHTML = `<div class="card">Erreur: ${err}</div>`;
    }
  }

  function renderStatus(data) {
    const container = document.getElementById('status');
    const cards = [];
    if (data.gpus) {
      for (const gpu of data.gpus) {
        cards.push(`
          <div class="card">
            <div style="font-weight:700;">GPU ${gpu.id} · ${gpu.name}</div>
            <div>Charge: ${(gpu.load * 100).toFixed(1)}%</div>
            <div>VRAM: ${(gpu.memory_used/1024).toFixed(1)} / ${(gpu.memory_total/1024).toFixed(1)} Go</div>
            <div>Température: ${gpu.temperature ?? 'N/A'}°C</div>
          </div>
        `);
      }
    }
    if (data.system) {
      cards.push(`
        <div class="card">
          <div style="font-weight:700;">Système</div>
          <div>CPU: ${data.system.cpu_percent}%</div>
          <div>Mémoire: ${data.system.memory_percent}%</div>
        </div>
      `);
    }
    container.innerHTML = cards.join('\n');
  }

  async function uploadAudio() {
    const fileInput = document.getElementById('file-input');
    const status = document.getElementById('upload-status');
    const button = document.getElementById('submit-audio');
    status.textContent = '';
    const file = fileInput.files[0];
    if (!file) { status.textContent = 'Choisissez un fichier audio.'; return; }

    button.disabled = true;
    status.textContent = 'En cours...';
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/process_audio', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      renderTranslations(data.translations);
      renderTimeline(data.timeline);
      document.getElementById('raw').textContent = JSON.stringify(data.raw, null, 2);
      status.textContent = 'Terminé';
    } catch (err) {
      status.textContent = 'Erreur : ' + err;
    } finally {
      button.disabled = false;
    }
  }

  function renderTranslations(translations) {
    if (!translations) return;
    const container = document.getElementById('translations');
    container.innerHTML = `
      <div class="grid">
        <div class="card"><h3>Anglais</h3><pre>${translations.english || ''}</pre></div>
        <div class="card"><h3>Français</h3><pre>${translations.french || ''}</pre></div>
        <div class="card"><h3>Allemand</h3><pre>${translations.german || ''}</pre></div>
      </div>
    `;
  }

  function formatTime(seconds) {
    if (isNaN(seconds)) return '—';
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  function renderTimeline(entries) {
    const container = document.getElementById('timeline');
    if (!entries || !entries.length) { container.innerHTML = '<p>Aucun segment diarizé.</p>'; return; }
    const rows = entries.map(e => `
      <tr>
        <td>${formatTime(e.start)}</td>
        <td>${formatTime(e.end)}</td>
        <td><strong>${e.role || e.speaker}</strong> <br/><span style="color:#8ba0b5;">(${e.speaker})</span></td>
        <td>${e.text || ''}</td>
      </tr>
    `);
    container.innerHTML = `
      <h3>Timeline</h3>
      <table>
        <thead><tr><th>Début</th><th>Fin</th><th>Locuteur</th><th>Texte</th></tr></thead>
        <tbody>${rows.join('')}</tbody>
      </table>
    `;
  }

  // Initialisation
  refreshStatus();
</script>
</body>
</html>
"""

INDEX_HTML = INDEX_HTML_TEMPLATE.replace("__API_BASE__", API_BASE)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
