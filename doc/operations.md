# Exploitation et opérations

Ce guide regroupe les bonnes pratiques pour déployer, exploiter et dépanner la passerelle d'inférence.

## Pré-requis matériels et logiciels

- GPU NVIDIA compatible CUDA 12.4+ (plateforme cible : DGX 8×H200).
- Pilotes NVIDIA et runtime `nvidia-container-toolkit` à jour.
- Accès réseau à Hugging Face (modèles Canary & Pyannote sont sous contrôle de jeton).
- Espace disque conséquent pour le cache (`/models`, plusieurs dizaines de Go).
- Python 3.10+ si exécution hors conteneur.

## Variables d'environnement clés

| Variable | Description | Défaut |
|----------|-------------|--------|
| `API_HOST` / `API_PORT` | Adresse et port de l'API FastAPI. | `0.0.0.0` / `8000` |
| `HUGGINGFACE_TOKEN` | Jeton HF pour télécharger les modèles protégés. | `null` |
| `MODEL_CACHE_DIR` | Répertoire partagé pour les artefacts téléchargés. | `/models` |
| `OPENAI_API_KEYS` | Liste de clés (séparées par virgules) autorisées pour l'API `/v1`. | `[]` |
| `LAZY_LOAD_MODELS` | Si `false`, charge tous les modèles au démarrage. | `true` |
| `FRONTEND_DIST` | Chemin vers le build statique du tableau de bord. | `/app/frontend` |
| `LOG_LEVEL` | Niveau de log (`debug`, `info`, `warning`, ...). | `info` |

Les scripts `build_docker.sh` et `run_docker.sh` acceptent en plus :

- `HOST_PORT`, `HOST_BIND_ADDRESS` : exposition réseau.
- `MODEL_CACHE_DIR` (bind mount hôte → conteneur).
- `ADVERTISED_HOSTS` : liste d'hôtes à afficher à l'issue du démarrage.

## Construction & déploiement Docker

1. **Construction** : `./build_docker.sh mon-image:latest`
   - Étape frontend : installe Node 20, build Vite.
   - Étape backend : image `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, installation Python + dépendances GPU.
2. **Exécution** : `./run_docker.sh mon-image:latest`
   - Monte `${MODEL_CACHE_DIR:-$(pwd)/model_cache}` sur `/models` dans le conteneur.
   - Passe le jeton Hugging Face et les variables d'environnement.
   - Rend l'intégralité des GPU visibles (`--gpus all`).

### Points d'attention

- Le build installe `apt-utils` pour éviter les erreurs de configuration `debconf`.
- `MODEL_CACHE_DIR` doit être persistant pour éviter les re-téléchargements.
- Le conteneur publie par défaut `8000/tcp`; ajuster `HOST_PORT` selon l'environnement.

## Gestion du cache Hugging Face

- Les modèles sont stockés dans `MODEL_CACHE_DIR` avec la convention `models--<org>--<repo>`.
- `BaseModelWrapper.cache_has_artifacts` valide la présence d'au moins un fichier.
- Pour préparer le cache hors ligne, utiliser le bouton **Télécharger** du tableau de bord ou l'endpoint `POST /api/admin/models/<clé>/download`.
- Les jetons sont persistés dans `MODEL_CACHE_DIR/.hf_token`.

## Gestion GPU

- `preferred_device_ids` peut être fourni lors du chargement (`POST /api/admin/models/qwen/load` avec `{ "gpu_device_ids": [1,2] }`).
- Qwen ajuste `CUDA_VISIBLE_DEVICES` avant d'initialiser vLLM et le restaure ensuite.
- Canary force `torch.cuda.set_device(<id>)` ; Pyannote déplace explicitement son pipeline via `pipeline.to(...)`.
- `GPUMonitor` expose mémoire utilisée, charge et température. Le front calcule les pourcentages pour afficher des badges.

## Journalisation & supervision

- Les logs applicatifs sont diffusés sur stdout **et** dans un fichier rotatif `LOG_DIR/LOG_FILE_NAME` (par défaut `/var/log/servertestllm/backend.log`).
- `ModelRegistry` trace désormais l'instanciation des wrappers, les changements de préférences GPU, les téléchargements forcés ainsi que l'arrêt propre lors du shutdown.
- `GPUMonitor` journalise les indisponibilités CUDA/GPUtil afin d'identifier rapidement un conteneur lancé sans GPU ou avec une installation incomplète.
- `ModelWrapper.update_runtime()` enregistre les actions importantes (téléchargement, chargement, erreurs) pour le tableau de bord.
- `dependency_inspector` informe des incompatibilités CUDA (ex. torchaudio CPU-only).

## Sécurité

- L'API `/v1/*` peut être restreinte via `OPENAI_API_KEYS`.
- Le tableau de bord ne persiste pas la clé si le navigateur refuse `localStorage` (mode privé).
- Les endpoints audio acceptent uniquement des fichiers `multipart/form-data`; prévoyez un proxy HTTPS en production.

## Dépannage rapide

| Symptôme | Causes probables | Actions |
|----------|------------------|---------|
| Téléchargement bloqué à 0 % | Token Hugging Face absent/incorrect | Vérifier `HUGGINGFACE_TOKEN`, le mettre à jour via l'UI admin. |
| Chargement modèle échoue avec « CUDA requis » | GPU non visible dans le conteneur | Vérifier `nvidia-smi`, `--gpus all`, variables `CUDA_VISIBLE_DEVICES`. |
| Pyannote ne trouve pas ses checkpoints | Cache incomplet ou références `$MODEL` non résolues | Forcer `POST /api/admin/models/pyannote/download`, vérifier l'espace disque. |
| API `/v1` renvoie 401/403 | Clé API non fournie ou invalide | Configurer la clé dans l'UI (dialogue en haut à droite) ou via header `Authorization`. |
| Tableau de bord vide | Frontend non servi ou `FRONTEND_DIST` incorrect | Regénérer le build (`npm run build`), vérifier le chemin monté. |

## Stratégie de mise à jour

1. Construire une nouvelle image (versionnée) via `build_docker.sh`.
2. Valider hors-ligne en compilant les modules Python : `python -m compileall backend/app` (fonctionne sur machine CPU).
3. Déployer sur un environnement de pré-production avec un cache dédié.
4. Purger les anciens artefacts si nécessaire (`rm -rf MODEL_CACHE_DIR/models--...`).
5. Surveiller les métriques GPU lors du premier chargement (vLLM peut utiliser >90 % de la VRAM configurée).

Pour les détails de l'architecture logicielle, consultez `doc/architecture.md`. Pour des informations sur les modules, voir `doc/backend.md` et `doc/frontend.md`.
