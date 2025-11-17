# Dashboard audio autonome (hors conteneur)

Ce script FastAPI minimaliste permet d'interroger l'API d'inférence exposée (Canary, Pyannote, OpenAI-compatible) sans passer par le conteneur principal. Il affiche l'état GPU et orchestre la pipeline audio (transcription → diarisation → traduction) depuis une simple page HTML.

## Fichier

- `scripts/standalone_audio_dashboard.py`

## Prérequis

- Python 3.10+ avec les dépendances :
  ```bash
  pip install fastapi uvicorn httpx python-multipart
  ```
- Accès réseau à l'API d'inférence (par défaut `http://10.200.50.45:8000`).
- Facultatif : un token d'autorisation si l'API l'exige.

## Variables d'environnement

- `UPSTREAM_API_BASE` (par défaut `http://10.200.50.45:8000`) : URL de la passerelle existante.
- `UPSTREAM_API_TOKEN` : jeton bearer à transmettre aux appels `/api/*` et `/v1/*` si nécessaire.
- `UPSTREAM_CHAT_MODEL` : modèle pour les traductions et la détection de rôles (par défaut `qwen`).

## Lancement

```bash
python scripts/standalone_audio_dashboard.py
# ou, pour choisir un port personnalisé
UPSTREAM_API_BASE=http://<hote>:<port> uvicorn scripts.standalone_audio_dashboard:app --host 0.0.0.0 --port 8100
```

Ouvrir ensuite `http://localhost:8100` dans un navigateur.

## Fonctionnement

1. **Statut GPU** : la page interroge `/status`, qui proxifie `/api/admin/status` pour récupérer charge/VRAM/CPU.
2. **Upload audio** : l'utilisateur choisit un fichier WAV/MP3/FLAC.
3. **Transcription** : le fichier est envoyé à `/api/audio/transcribe` (Canary). Le texte est normalisé en anglais via le modèle chat, puis traduit en français et allemand.
4. **Diarisation** : le même fichier est envoyé à `/api/diarization/process` (Pyannote). Les segments sont enrichis avec une étiquette de rôle contextuelle (journaliste, médecin, patient, président, etc.) inférée par le modèle chat.
5. **Affichage** : la timeline liste chaque segment avec timestamps, rôle et texte. Les traductions s'affichent en trois colonnes, et un panneau "Brut" montre les réponses JSON pour debug.

## Points d'attention

- Le script ne stocke rien en disque ; les fichiers sont relayés en mémoire et supprimés après traitement.
- Le prompt de labellisation des locuteurs est volontairement concis pour obtenir des étiquettes courtes et cohérentes.
- Le frontend est volontairement léger (HTML/CSS/JS natif) pour être déployé n'importe où sans Node/npm.
