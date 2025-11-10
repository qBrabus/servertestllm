# Backend FastAPI

Le backend constitue le cœur de la passerelle. Il est structuré autour d'une application **FastAPI** déployée via `uvicorn` et organisée en modules clairs (`config`, `routers`, `services`, `models`, `schemas`).

## Configuration (`config.py`)

- **`Settings`** : configuration Pydantic (hôte, port, niveau de log, cache modèle, origine CORS, clés API, etc.).
- Les valeurs sont chargées depuis les variables d'environnement (et facultativement un `.env`).
- Les clés API OpenAI peuvent être fournies sous forme de liste séparée par des virgules (`OPENAI_API_KEYS`).
- `model_cache_dir` et `frontend_dist` acceptent des chemins absolus ou relatifs.

## Cycle de vie de l'application (`main.py`)

1. Création de l'application FastAPI + configuration CORS.
2. Montage du frontend statique si `FRONTEND_DIST` pointe vers un build Vite valide.
3. **Événement de démarrage** :
   - initialisation du logger (niveau configurable via `LOG_LEVEL`),
   - configuration du `TokenStore` et propagation du jeton Hugging Face,
   - initialisation du `ModelRegistry` et du `GPUMonitor`,
   - pré-chargement des modèles si `LAZY_LOAD_MODELS=false`.
4. **Événement d'arrêt** : arrêt propre du registre (déchargement VRAM) et du monitor GPU.
5. **Gestion globale des exceptions** : renvoie un JSON 500 documentant l'erreur, avec log côté serveur.

## Routes principales

| Route | Module | Description |
|-------|--------|-------------|
| `/v1/chat/completions`, `/v1/completions` | `routers/openai.py` | API compatible OpenAI (chat et completions). Vérifie la clé API si configurée, normalise le nom du modèle et délègue au wrapper Qwen. |
| `/api/audio/transcribe` | `routers/audio.py` | Réception d'un fichier audio (`multipart/form-data`), inférence via Canary ASR, retourne transcription + fréquence cible. |
| `/api/diarization/process` | `routers/diarization.py` | Diarisation Pyannote sur GPU, réponse JSON avec segments (speaker, start, end). |
| `/api/admin/status` | `routers/admin.py` | Aggrège l'état du système pour le tableau de bord (GPU, métriques, modèles, dépendances). |
| `/api/admin/models/{clé}/(download|load|unload)` | `routers/admin.py` | Actions directes sur le registre (téléchargement, chargement, déchargement). |
| `/api/admin/huggingface/token` | `routers/admin.py` | Lecture/écriture du jeton Hugging Face, persisté par `TokenStore`. |

## Services transverses

### Registre de modèles (`services/model_registry.py`)

- Maintient un dictionnaire `ModelSlot` (métadonnées + fabrique + instance). Les clés par défaut : `qwen`, `canary`, `pyannote`.
- `configure()` crée le cache si absent, enregistre le token et prépare les fabriques.
- `ensure_loaded()` instancie et charge le modèle, avec gestion des préférences GPU.
- `ensure_downloaded()` ne fait que synchroniser les artefacts (utile hors ligne).
- `status()` compile un résumé `ModelStatus` incluant la progression, le serveur exposé, les erreurs éventuelles et la disponibilité du cache.

### Stockage du token (`services/token_store.py`)

- Persistance simple sur disque (`/models/.hf_token`).
- Accès thread-safe via un verrou.
- Utilisé au démarrage pour restaurer un jeton existant ou en enregistrer un nouveau.

### Supervision GPU (`services/gpu_monitor.py`)

- Thread en tâche de fond déclenché toutes les 5 s.
- Utilise `GPUtil` pour relever nom, mémoire utilisée et température GPU.
- Complété par `psutil` pour la charge CPU/RAM globale.

### Inspection des dépendances (`services/dependency_inspector.py`)

- Vérifie la présence de `torch`, `torchvision`, `torchaudio`.
- Rapporte la version, l'état CUDA (runtime compilé, extensions disponibles) et remonte les erreurs de configuration.

## Wrappers de modèles (`backend/app/models`)

Tous héritent de `BaseModelWrapper` qui gère :

- le verrouillage asynchrone (`asyncio.Lock`) pour éviter les chargements concurrents ;
- la mise à jour atomique de l'état runtime (progression, messages, erreurs, métadonnées serveur) exposé au tableau de bord ;
- les helpers de téléchargement Hugging Face avec barre de progression ;
- la normalisation des chemins de cache (`models--<repo>`).

### Qwen (`models/qwen.py`)

- Télécharge `Qwen/Qwen3-VL-30B-A3B-Instruct` et instancie `AsyncLLMEngine` (vLLM) + tokenizer.
- Gère `CUDA_VISIBLE_DEVICES` pour appliquer les préférences GPU.
- Utilise `SamplingParams` (max_tokens, température, top_p) et la template chat de Qwen.
- Vide la VRAM (`torch.cuda.empty_cache()`) et restaure `CUDA_VISIBLE_DEVICES` au déchargement.

### Canary ASR (`models/canary.py`)

- Synchronise le checkpoint `canary-1b-v2.nemo`.
- Restaure le modèle avec `ASRModel.restore_from` (NeMo) sur le GPU primaire.
- Convertit l'audio en mono float32, le rééchantillonne à 16 kHz via `torchaudio.functional.resample` et utilise un fichier temporaire `.wav` pour la transcription.

### Pyannote (`models/pyannote_model.py`)

- Télécharge les artefacts Pyannote (poids `.bin/.ckpt`, config `.yaml/.json`).
- Patche le pipeline pour résoudre les références `$MODEL/...` en chemins locaux.
- Vérifie `pyannote.audio>=4` et la présence CUDA.
- Déplace explicitement le pipeline sur `cuda:<gpu>` choisi.
- Produit une liste de segments (speaker, début, fin).

## Schémas Pydantic (`backend/app/schemas`)

- `openai.py` : structures `ChatCompletionRequest/Response`, `CompletionRequest/Response` et `UsageInfo`.
- `audio.py` : réponses pour la transcription (`text`, `sampling_rate`) et la diarisation (segments).
- `admin.py` : objets pour l'état du tableau de bord, la description des GPU, modèles, dépendances et payloads de chargement.

## Bonnes pratiques d'extension

1. **Ajouter un nouveau modèle** :
   - créer un wrapper héritant de `BaseModelWrapper`,
   - l'enregistrer via `ModelRegistry.register()` (éventuellement après un `configure(..., with_defaults=False)` pour composer une configuration personnalisée),
   - mettre à jour le frontend si nécessaire (nouvelle carte, actions).
2. **Nouvelles routes** : centraliser les schémas Pydantic dans `schemas/`, réutiliser `ModelRegistry` pour la gestion du cycle de vie.
3. **Instrumentation** : utiliser `update_runtime()` pour informer le tableau de bord (progression, erreurs) et fournir un `server` avec métadonnées utiles.
4. **Environnements CPU-only** : ne sont pas supportés pour l'inférence ; utiliser `python -m compileall backend/app` pour une validation statique.

## Dépendances critiques

- **PyTorch 2.6.0 + CUDA 12.4** : défini via `--index-url` dans `backend/requirements.txt`.
- **vLLM 0.11.0**, **NeMo 2.1.0**, **pyannote.audio 4.0.1**.
- `gputil`, `psutil` pour la supervision.
- Les épingles strictes (`numpy`, `scipy`, `pandas`, etc.) évitent les re-résolutions pip coûteuses.

Pour des scénarios d'exploitation avancés (mise en cache hors ligne, durcissement sécurité, tuning GPU), reportez-vous à `doc/operations.md`.
