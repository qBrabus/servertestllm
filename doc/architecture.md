# Architecture générale

Cette passerelle d'inférence unifiée orchestre trois familles de services IA dans un seul conteneur GPU :

- un backend **FastAPI** qui expose l'API REST et gère le cycle de vie des modèles ;
- des **wrappers de modèles** spécialisés (LLM, ASR, diarisation) pilotés par un registre commun ;
- un **tableau de bord React** permettant de suivre l'état des charges, de lancer des téléchargements et de piloter les inférences.

L'ensemble est conçu pour fonctionner en environnement à forte contrainte GPU (DGX 8×H200, CUDA >= 12.4) et suppose un accès Hugging Face pour télécharger les artefacts propriétaires.

## Vue d'ensemble

```text
         ┌──────────────────────────────────────────────────────────────┐
         │                          Conteneur Docker                     │
         │                                                              │
         │  ┌─────────────┐      ┌──────────────────────────┐           │
         │  │  Frontend   │◀────▶│  API FastAPI (/api, /v1) │◀────────┐ │
         │  │  (Vite+MUI) │ Web  │                          │         │ │
         │  └─────────────┘      └───────────────┬──────────┘         │ │
         │                                       │                    │ │
         │                             ┌─────────▼─────────┐          │ │
         │                             │  Registre modèle   │          │ │
         │                             │  (ModelRegistry)   │          │ │
         │                             └─────────┬─────────┘          │ │
         │                                       │                    │ │
         │                     ┌─────────────────┼─────────────────┐  │ │
         │                     │                 │                 │  │ │
         │          ┌──────────▼──────┐  ┌───────▼─────────┐  ┌─────▼──────────┐
         │          │  Qwen 30B (LLM) │  │ Canary ASR (.nemo│  │ Pyannote diar. │
         │          │  vLLM Async     │  │  NeMo)           │  │  Pipeline      │
         │          └─────────────────┘  └──────────────────┘  └────────────────┘
         │                                                              │
         │                    ┌────────────────────────────────┐        │
         │                    │  Monitor GPU / dépendances     │◀───────┘
         │                    └────────────────────────────────┘
         └──────────────────────────────────────────────────────────────┘
```

### Points clés

- **Registre de modèles** : `ModelRegistry` instancie et met en cache chaque wrapper (`QwenModel`, `CanaryASRModel`, `PyannoteDiarizationModel`). Il gère le téléchargement, le chargement sur GPU et la libération de VRAM.
- **Gestion des tokens** : `TokenStore` persiste le jeton Hugging Face sur disque (`/models/.hf_token`) et le propage aux wrappers en mémoire.
- **Supervision** : `GPUMonitor` relève périodiquement l'état des GPU et des ressources systèmes. `dependency_inspector` vérifie que Torch/Torchaudio/Torchvision sont bien compilés avec CUDA.
- **Front-end** : construit avec Vite + React + Material UI, il consomme les routes `/api/admin/*`, `/api/audio/*`, `/api/diarization/*` et `/v1/*` pour présenter un tableau de bord temps réel.

## Flux principaux

### 1. Chat completions (API OpenAI compatible)

1. Le client envoie `POST /v1/chat/completions` avec la liste de messages.
2. Le routeur `routers/openai.py` résout le modèle (`qwen`).
3. `ModelRegistry.get` renvoie (ou instancie) le wrapper `QwenModel`.
4. `QwenModel.ensure_loaded()` s'assure que vLLM est initialisé, télécharge les poids si nécessaire, puis interroge `AsyncLLMEngine.generate`.
5. La réponse formatée est renvoyée avec les statistiques d'utilisation (`prompt_tokens`, `completion_tokens`).

### 2. Transcription audio

1. Le client envoie un fichier audio via `POST /api/audio/transcribe`.
2. `CanaryASRModel.ensure_loaded()` restaure le checkpoint `.nemo` via NeMo Toolkit et transforme l'audio en tenseurs PyTorch (`torchaudio.functional.resample`).
3. La transcription est retournée avec la fréquence d'échantillonnage cible (16 kHz par défaut).

### 3. Diarisation

1. Le client poste un flux audio à `POST /api/diarization/process`.
2. `PyannoteDiarizationModel` garantit la présence des artefacts Pyannote, patch la résolution des chemins (`$MODEL/...`) et exécute le pipeline sur GPU.
3. La liste des segments (speaker, start, end) est renvoyée.

### 4. Supervision du tableau de bord

- `GET /api/admin/status` agrège l'état des GPU, des métriques système, des modèles et des dépendances. Les wrappers exposent un `runtime_status` mis à jour à chaque étape (téléchargement, chargement, erreur).
- Les actions de téléchargement/chargement/déchargement (`POST /api/admin/models/<clé>/*`) appellent directement les méthodes du registre.

## Dossiers logiques

| Composant | Chemin | Description |
|-----------|--------|-------------|
| Backend FastAPI | `backend/app` | Configuration (`config.py`), point d'entrée (`main.py`), routes, schémas Pydantic, services transverses. |
| Wrappers modèle | `backend/app/models` | Implémentations `BaseModelWrapper` pour Qwen, Canary, Pyannote. |
| Frontend React | `frontend/src` | Pages, composants et hooks pour le tableau de bord. |
| Scripts conteneur | `Dockerfile`, `build_docker.sh`, `run_docker.sh` | Construction et exécution GPU-ready. |
| Documentation | `doc/` | Notes techniques détaillées (architecture, backend, frontend, exploitation). |

Pour une exploration approfondie de chaque brique, reportez-vous aux fichiers dédiés dans ce dossier de documentation.
