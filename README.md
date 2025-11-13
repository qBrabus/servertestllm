# Passerelle d'inférence unifiée

Cette plateforme regroupe, dans un seul déploiement GPU, trois services d'IA complémentaires :

- **Qwen3 VL 30B** pour les conversations multimodales (servi par vLLM) ;
- **NVIDIA Canary** pour la transcription automatique multilingue ;
- **Pyannote** pour la diarisation des locuteurs.

L'API FastAPI expose des endpoints compatibles OpenAI ainsi que des routes audio spécialisées, tandis qu'une interface React offre une supervision temps réel (progression des téléchargements, état VRAM, dépendances CUDA). Toute la documentation et les messages sont en français afin de faciliter la maintenance interne.

## Table des matières

1. [Architecture](#architecture)
2. [Fonctionnalités principales](#fonctionnalités-principales)
3. [Prérequis](#prérequis)
4. [Structure du dépôt](#structure-du-dépôt)
5. [Installation locale](#installation-locale)
6. [Construction et exécution Docker](#construction-et-exécution-docker)
7. [Utilisation des APIs](#utilisation-des-apis)
8. [Supervision & tableau de bord](#supervision--tableau-de-bord)
9. [Tests et vérifications](#tests-et-vérifications)
10. [Mise à jour des dépendances](#mise-à-jour-des-dépendances)
11. [Documentation additionnelle](#documentation-additionnelle)

## Architecture

Le système s'articule autour de trois briques :

- un **backend FastAPI** (`backend/app`) qui gère le cycle de vie des modèles, les téléchargements Hugging Face et les routes REST ;
- des **wrappers de modèles** héritant d'un socle commun (`BaseModelWrapper`) pour orchestrer vLLM, NeMo et Pyannote ;
- un **tableau de bord React** (Vite + Material UI) servi en statique depuis FastAPI.

Un registre central (`ModelRegistry`) maintient la liste des modèles, applique les préférences GPU et expose l'état des opérations au front. `GPUMonitor` collecte la télémétrie (charge, VRAM, température), tandis que `dependency_inspector` vérifie la cohérence de la pile CUDA. Un schéma détaillé et les flux d'appels sont disponibles dans [`doc/architecture.md`](doc/architecture.md).

## Fonctionnalités principales

- **API OpenAI-compatible** : `POST /v1/chat/completions` et `POST /v1/completions` avec calcul des tokens consommés.
- **Transcription audio** : `POST /api/audio/transcribe` (Canary ASR) avec rééchantillonnage automatique.
- **Diarisation** : `POST /api/diarization/process` (Pyannote) renvoyant les segments (speaker, start, end).
- **Administration** : `GET /api/admin/status` pour l'état global, `POST /api/admin/models/{clé}/(download|load|unload)` pour piloter chaque modèle, `POST /api/admin/huggingface/token` pour gérer le jeton HF.
- **Interface web** : suivi temps réel des téléchargements, affichage des endpoints exposés, gestion des clés API, actions rapides.
- **Suivi GPU** : collecte continue de l'utilisation mémoire, de la charge et de la température sur chaque carte.

## Prérequis

- **Matériel** : GPU NVIDIA compatible CUDA ≥ 12.4 (plateforme cible : DGX 8×H200). La plate-forme est validée en production avec des H200 80 Go ; toute carte disposant d'au moins 70 Go de VRAM libre fonctionnera pour Qwen.
- **Pilotes** : driver ≥ 550.54.15 et `nvidia-container-toolkit` ≥ 1.16.2 pour exposer correctement CUDA 12.4 dans Docker.
- **Vérification rapide** :
  ```bash
  nvidia-smi | head -n 1
  # NVIDIA-SMI 555.85    Driver Version: 555.85    CUDA Version: 12.4
  ```
  Dans le conteneur, la commande suivante doit retourner `True` et `12.4` :
  ```bash
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
  ```
- **Réseau** : accès à Hugging Face avec un jeton `HUGGINGFACE_TOKEN` autorisant `nvidia/canary-1b-v2` et `pyannote/speaker-diarization-community-1`.
- **Logiciel (mode local)** : Python 3.10+, `pip`, `virtualenv`.

## Structure du dépôt

```text
.
├── backend/              # Application FastAPI (routes, services, modèles, schémas)
├── frontend/             # Tableau de bord React (Vite, Material UI)
├── doc/                  # Documentation technique détaillée
├── Dockerfile            # Construction multi-étapes GPU-ready
├── build_docker.sh       # Build script (frontend -> backend)
├── run_docker.sh         # Script de lancement avec GPU & cache modèles
└── README.md             # Ce document
```

Les dossiers `backend/app/models` et `backend/app/services` sont décrits en profondeur dans [`doc/backend.md`](doc/backend.md). Le frontend est détaillé dans [`doc/frontend.md`](doc/frontend.md).

## Installation locale

> ⚠️ Tous les modèles exigent un GPU CUDA visible. Les commandes suivantes ne suffisent pas en environnement CPU-only.

1. Créer l'environnement virtuel Python 3.10 :
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. Installer les dépendances backend (URLs CUDA incluses dans `requirements.txt`) :
   ```bash
   pip install -r backend/requirements.txt
   ```
   > ℹ️ Ce fichier est épinglé sur l'index PyTorch CUDA 12.4 et installe `torch==2.6.0+cu124`, `torchaudio==2.6.0+cu124` et `torchvision==0.21.0+cu124`. Aucune étape manuelle d'installation CUDA n'est requise tant que les pilotes hôtes sont compatibles.
3. Exporter les variables minimales :
   ```bash
   export HUGGINGFACE_TOKEN="<votre_token>"
   export MODEL_CACHE_DIR="$(pwd)/model_cache"
   export OPENAI_API_KEYS="cle1,cle2"   # optionnel
   ```
4. Lancer l'API FastAPI :
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
5. (Optionnel) Servir le frontend en développement :
   ```bash
   cd frontend
   npm install
   npm run dev -- --host
   ```
  Configurer le proxy Vite (`vite.config.ts`) pour rediriger `/api` et `/v1` vers `http://localhost:8000`.
6. Vérifier l'environnement CUDA effectif :
   ```bash
   curl -s http://localhost:8000/api/admin/status \
     | jq '.dependencies[] | select(.name=="torch")'
   # {
   #   "name": "torch",
   #   "version": "2.6.0+cu124",
   #   "cuda": true,
   #   "details": {
   #     "cuda_runtime": "12.4",
   #     "cudnn": "9.1.0",
   #     "cuda_available": true
   #   }
   # }
   ```

## Construction et exécution Docker

### Build

Le script [`build_docker.sh`](build_docker.sh) orchestre la construction multi-étapes (frontend puis backend) :
```bash
./build_docker.sh mon-image:latest
```

- Base : `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- Étape frontend : Node 20 + build Vite
- Étape backend : installation Python, vLLM, NeMo, Pyannote, GPUtil, etc.
- `apt-utils` est installé en amont pour éviter l'erreur `debconf: delaying package configuration`.

### Run

```bash
./run_docker.sh mon-image:latest
```

Options principales (exporter avant exécution) :

| Variable | Rôle |
|----------|------|
| `HUGGINGFACE_TOKEN` | Jeton HF obligatoire pour Canary & Pyannote. |
| `OPENAI_API_KEYS` | Liste de clés autorisées pour `/v1/*` (sinon accès libre). |
| `MODEL_CACHE_DIR` | Dossier hôte monté sur `/models` (cache persistant recommandé). |
| `HOST_PORT` | Port hôte mappé sur `8000` (défaut : `8000`). |
| `HOST_BIND_ADDRESS` | Adresse d'écoute côté hôte (permet de restreindre la publication). |
| `ADVERTISED_HOSTS` | Liste d'adresses à afficher une fois le conteneur démarré. |

Le script rend tous les GPU visibles (`--gpus all`) et affiche automatiquement l'URL d'accès (`http://<adresse>:<port>`), en filtrant les IP Docker internes pour éviter la confusion.

#### Contrôles après démarrage

- `docker logs <container>` doit afficher la bannière FastAPI suivie des messages `ModelRegistry configured` et `GPUMonitor started`.
- `docker exec -it <container> python -c "import torch; print(torch.version.cuda)"` → `12.4`.
- `curl http://<hôte>:<port>/api/admin/status | jq '.gpu.devices[] | {name, total, used}'` pour confirmer la visibilité des GPU.

## Utilisation des APIs

### Endpoints principaux

| Endpoint | Description | Modèle cible |
|----------|-------------|---------------|
| `POST /v1/chat/completions` | Chat completion compatible OpenAI. | Qwen3 VL 30B |
| `POST /v1/completions` | Complétion texte simple. | Qwen3 VL 30B |
| `POST /api/audio/transcribe` | Transcription audio (fichier `multipart/form-data`). | NVIDIA Canary |
| `POST /api/diarization/process` | Diarisation audio (segments JSON). | Pyannote |
| `GET /api/admin/status` | État global (GPU, dépendances, modèles). | Tous |
| `POST /api/admin/models/{clé}/download` | Pré-télécharge les artefacts HF. | Tous |
| `POST /api/admin/models/{clé}/load` | Charge le modèle sur GPU (option `gpu_device_ids`). | Tous |
| `POST /api/admin/models/{clé}/unload` | Libère la VRAM. | Tous |
| `POST /api/admin/huggingface/token` | Met à jour/persiste le jeton HF. | N/A |

#### Exemples d'appel

- **Chat completions** :
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <clé_si_configurée>" \
    -d '{
      "model": "qwen",
      "messages": [
        {"role": "system", "content": "Tu es un assistant concis."},
        {"role": "user", "content": "Présente la pile CUDA utilisée."}
      ]
    }'
  ```
- **Transcription audio** :
  ```bash
  curl -X POST http://localhost:8000/api/audio/transcribe \
    -H "Authorization: Bearer <clé>" \
    -F "file=@sample.wav"
  ```

Les clés valides pour `{clé}` sont `qwen`, `canary`, `pyannote`. Le détail de chaque wrapper (paramètres, contraintes) est documenté dans [`doc/modeles.md`](doc/modeles.md).

### Gestion des clés API `/v1`

- Définir `OPENAI_API_KEYS="cle1,cle2"` côté serveur.
- Le frontend propose un dialogue (icône cadenas dans la barre supérieure) pour enregistrer la clé côté navigateur.
- Les requêtes clients doivent fournir `Authorization: Bearer <clé>`.

## Supervision & tableau de bord

- Accès par défaut : `http://<hôte>:<port>/`
- Cartes modèles : progression du téléchargement, état (`idle`, `loading`, `ready`, `error`), actions Télécharger/Charger/Décharger.
- Panneau latéral : GPU (nom, VRAM utilisée/total, température), métriques système (CPU/RAM), dépendances CUDA (torch, torchvision, torchaudio).
- Les endpoints exposés sont listés avec l'URL complète, les raccourcis (copier) et la documentation OpenAPI (`/docs`).
- `useDashboard` adapte l'intervalle de rafraîchissement : 1 s pendant les chargements, 5 s au repos.
- Un fichier de log rotatif est produit dans `LOG_DIR` (défaut : `/var/log/servertestllm/backend.log`) en plus de la sortie standard. Les messages du `ModelRegistry` et du `GPUMonitor` y consignent les actions (instanciation, téléchargement, indisponibilité GPU).

Pour une description complète des composants React et des flux de données, se référer à [`doc/frontend.md`](doc/frontend.md).

## Tests et vérifications

- **Validation statique** (environnement CPU) :
  ```bash
  python -m compileall backend/app
  ```
- **Smoke test CUDA** (nécessite un GPU) :
  ```bash
  python - <<'PY'
  import torch
  print('cuda_available=', torch.cuda.is_available())
  print('cuda_version=', torch.version.cuda)
  print('device_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
  PY
  ```
- **Lint/TypeScript frontend** :
  ```bash
  cd frontend
  npm run lint
  npm run typecheck
  ```
- **Tests unitaires backend** :
  ```bash
  python -m unittest backend.tests.test_model_registry
  ```
- **Contrôle des dépendances GPU** : appeler `GET /api/admin/status` et vérifier la section `dependencies` (doit indiquer `cuda: true`).

L'exécution d'inférences effectives nécessite un GPU CUDA disponible. En CI CPU-only, se limiter à la compilation et aux linters.

## Mise à jour des dépendances

Pour rafraîchir les épingles Python et valider la compatibilité CUDA, utiliser [uv](https://github.com/astral-sh/uv) :

```bash
uv pip compile backend/requirements.in --python-version 3.12 --resolution=highest --output-file backend/requirements.txt
```

Cette commande régénère `backend/requirements.txt` en respectant l'index PyTorch CUDA 12.4. Elle évite les conflits de type `nvidia-curand`/`torch` rencontrés lors des installations Docker.

## Documentation additionnelle

- [`doc/architecture.md`](doc/architecture.md) : schémas, flux d'appels et vue d'ensemble du système.
- [`doc/backend.md`](doc/backend.md) : analyse des modules FastAPI, services et wrappers.
- [`doc/frontend.md`](doc/frontend.md) : structure du tableau de bord, interactions avec l'API.
- [`doc/operations.md`](doc/operations.md) : bonnes pratiques de déploiement, gestion du cache, dépannage rapide.
- [`doc/modeles.md`](doc/modeles.md) : fiches techniques des modèles (contraintes GPU, taille, particularités).

Bon usage ! Toute contribution doit conserver l'exécution GPU-only, les épingles de dépendances et la documentation en français.
