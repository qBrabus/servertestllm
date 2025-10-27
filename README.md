# Passerelle d'Inférence Unifiée

Ce projet réunit dans un même conteneur GPU une API de conversation OpenAI-compatible, un service de reconnaissance vocale Canary (.nemo), ainsi qu'un pipeline de diarisation Pyannote. L'ensemble du code et de la documentation est désormais en français pour faciliter la maintenance de l'équipe. Toutes les inférences sont exécutées sur GPU (DGX 8xH200, CUDA >= 12.4) – aucun retour n'est prévu sur CPU en dehors d'opérations annexes inévitables (lecture disque, pré/post-traitements légers).

## Fonctionnalités principales

- **LLM Qwen3 VL 30B** servi par [vLLM](https://github.com/vllm-project/vllm) pour des réponses rapides et stables.
- **ASR multilingue NVIDIA Canary** (`.nemo`) via [NeMo Toolkit](https://github.com/NVIDIA/NeMo) avec exécution exclusivement sur GPU.
- **Diarisation Pyannote** avec transfert systématique du pipeline sur GPU.
- **API OpenAI-compatible** : `POST /v1/chat/completions` et `POST /v1/completions`.
- **API audio** : `POST /api/audio/transcribe` (transcription) et `POST /api/diarization/process` (diarisation).
- **Tableau de bord React** (Vite + Material UI) pour surveiller l'état des modèles, l'utilisation GPU et exécuter des requêtes.

## Prérequis

- Python 3.10 (testé) si vous lancez le backend hors conteneur.
- Docker 24+ avec le runtime NVIDIA ou `nvidia-docker2` pour l'exécution conteneurisée.
- Pilotes NVIDIA compatibles CUDA 12.4 et accès à un serveur DGX 8xH200.
- Un jeton Hugging Face (`HUGGINGFACE_TOKEN`) autorisant le téléchargement des modèles Canary et Pyannote.
- Accès réseau à Hugging Face (les modèles sont tirés dynamiquement).

## Installation locale (hors Docker)

1. Créez un environnement virtuel Python 3.10 :
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. Installez les dépendances backend (les URLs nécessaires pour les roues CUDA sont intégrées au fichier) :
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Exportez les variables requises (au minimum le jeton Hugging Face) :
   ```bash
   export HUGGINGFACE_TOKEN="<token>"
   export MODEL_CACHE_DIR="$(pwd)/model_cache"
   ```
4. Lancez l'API FastAPI en local :
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

> **Important** : Les modèles refuseront de se charger si aucun GPU n'est visible (`CUDA_VISIBLE_DEVICES`). Pensez à vérifier `nvidia-smi` avant de démarrer.

## Construction et exécution Docker

### Construction

Le script `build_docker.sh` orchestre la construction multi-étapes (frontend puis backend) :

```bash
./build_docker.sh mon-image:latest
```

Le `Dockerfile` s'appuie sur `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, installe Node 20 pour le frontend, puis FastAPI/vLLM/NeMo/Pyannote côté backend avec les roues CUDA fournies par NVIDIA. L'étape système installe désormais `apt-utils` en amont afin d'éviter l'avertissement `debconf: delaying package configuration` lors des builds.

### Exécution

Le script suivant publie le port 8000, partage le cache de modèles et rend tous les GPU visibles :

```bash
./run_docker.sh mon-image:latest
```

Variables d'environnement principales (à exporter avant l'exécution si nécessaire) :

- `HUGGINGFACE_TOKEN` : jeton Hugging Face obligatoire pour Canary et Pyannote.
- `OPENAI_KEYS` : chaîne de clés API séparées par des virgules pour sécuriser les routes OpenAI (facultatif, accès libre sinon).
- `MODEL_CACHE_DIR` : répertoire hôte pour la mise en cache (`./model_cache` par défaut).
- `HOST_PORT` : port hôte mappé sur `8000` (par défaut `8000`).
- `HOST_BIND_ADDRESS` : adresse d'écoute côté hôte passée à `docker -p`. Permet par exemple de forcer
  la publication uniquement sur `10.200.50.46` si l'accès doit rester sur cette IP.
- `ADVERTISED_HOSTS` : liste d'adresses (séparées par des virgules ou des espaces) à afficher après
  le démarrage du conteneur. Utile lorsque le serveur doit rester joignable via une IP virtuelle ou
  un alias qui n'est pas détecté automatiquement.

À défaut d'une liste personnalisée, `run_docker.sh` détecte désormais l'adresse source utilisée pour
la route par défaut (via `ip route get`). Les adresses internes propres à Docker (`172.17.x.x`) et
`localhost` ne sont plus affichées automatiquement pour éviter toute confusion lorsqu'on expose le
service sur un réseau privé.

## Utilisation

- **Tableau de bord** : `http://<hôte>:<port>/`
- **API OpenAI** : `http://<hôte>:<port>/v1/chat/completions` et `/v1/completions`
- **Transcription audio** : `http://<hôte>:<port>/api/audio/transcribe`
- **Diarisation** : `http://<hôte>:<port>/api/diarization/process`
- **Administration** : `GET /api/admin/status`, `POST /api/admin/models/{clé}/download`, `POST /api/admin/models/{clé}/load`, `POST /api/admin/models/{clé}/unload`

Clés modèles disponibles :

- `qwen` — LLM multimodal (vLLM)
- `canary` — ASR NeMo
- `pyannote` — Diarisation

Chaque carte du tableau de bord propose désormais trois actions distinctes :

- **Télécharger** : récupère les poids Hugging Face sans initialiser le runtime (utile pour préparer le cache hors ligne).
- **Charger** : initialise le modèle sur GPU en réutilisant les poids présents dans le cache local.
- **Décharger** : libère la mémoire GPU occupée par le modèle.

### Tableau de bord (mise à jour octobre 2025)

- Les cartes modèles affichent une barre de progression synchronisée en direct avec le téléchargement Hugging Face (la taille
  totale du dépôt est estimée et suivie pour refléter l'état réel entre 0 et 100%).
- Le badge « Téléchargement requis » bascule automatiquement en « Artefacts en cache » dès que les fichiers sont présents sur le
  disque, y compris lorsqu'un modèle reste à l'état *idle*.
- Les points d'accès exposés affichent désormais l'hôte, le port, l'URL complète, ainsi que des raccourcis pour copier les
  endpoints REST (`/api/audio/transcribe`, `/api/diarization/process`, `/v1/chat/completions`, etc.).
- Le panneau latéral renseigne l'utilisation CPU/RAM du serveur, les GPU disponibles et les informations de cache global
  (modèles prêts / artefacts téléchargés).
- Les cartes gagnent une mise en page responsive (3 colonnes en écran large, 2 sur portable, 1 sur mobile) avec un design
  modernisé : fond en dégradé, actions compactes, sélecteur GPU enrichi.
- Le pipeline Pyannote détecte correctement les artefacts locaux et charge le modèle via son identifiant Hugging Face, ce qui
  supprime l'erreur `Repo id must be in the form 'repo_name'...` lors du chargement.

> **Astuce navigateur** : le tableau de bord peut fonctionner en contexte « non sécurisé » (HTTP
> sur IP privée). Lorsque le stockage local est indisponible (Firefox mode strict, Safari privé), la
> clé API OpenAI-compatible n'est simplement pas mémorisée d'une session à l'autre au lieu de
> provoquer une page blanche.

La sélection des GPU a été améliorée : la liste déroulante affiche maintenant des cases à cocher pour visualiser rapidement les cartes choisies et permet de conserver le menu ouvert pour sélectionner plusieurs GPU d'affilée.

## Notes techniques

- Le répertoire `/models` (ou la valeur de `MODEL_CACHE_DIR`) est partagé entre tous les services et persiste entre deux démarrages pour éviter de re-télécharger les poids.
- Les scripts d'inférence convertissent systématiquement les entrées audio en tenseurs PyTorch et utilisent `torchaudio` pour le resampling, supprimant toute dépendance à Librosa/Numba.
- Qwen est chargé via `AsyncLLMEngine` de vLLM avec `tensor_parallel_size=1` par défaut ; le registre de modèles autorise toutefois la mise à jour dynamique des préférences GPU.
- Chaque wrapper vérifie explicitement la disponibilité CUDA et déclenche une erreur claire si aucun GPU n'est visible.
- Les dépendances Python critiques sont figées (`backend/requirements.txt`) pour éviter les erreurs `resolution-too-deep` de pip.

## Tests et maintenance

- Le backend suit une structure FastAPI classique (`backend/app`). Le point d'entrée est `app.main:app`.
- Pour vérifier la validité du code Python sans lancer les modèles (notamment en CI CPU-only), vous pouvez exécuter :
  ```bash
  python -m compileall backend/app
  ```
- Les modèles ne sont chargés que sur demande (lazy loading). Vous pouvez forcer le chargement au démarrage via `LAZY_LOAD_MODELS=false`.

Bon usage ! Toute contribution doit conserver ces contraintes GPU et la documentation en français.
