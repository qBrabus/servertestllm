# Détails techniques des modèles

Cette note approfondit la configuration et les contraintes des trois modèles embarqués.

## Qwen3 VL 30B (LLM multimodal)

- **Identifiant Hugging Face** : `Qwen/Qwen3-VL-30B-A3B-Instruct`.
- **Taille** : ~62 Go (poids + tokenizer) – prévoir autant d'espace disque et >70 Go de VRAM pour une exécution confortable.
- **Runtime** : [vLLM](https://github.com/vllm-project/vllm) en mode `AsyncLLMEngine`.
- **Paramètres par défaut** :
  - `tensor_parallel_size` : calculé à partir des GPU sélectionnés (`preferred_device_ids`).
  - `gpu_memory_utilization` : 0.90 (vLLM alloue jusqu'à 90 % de la VRAM disponible).
  - Longueur contexte maximale : 8192 tokens.
- **Mise en cache** :
  - Téléchargement via `BaseModelWrapper.download_snapshot` avec suivi détaillé de la progression.
  - Le tokenizer et le modèle partagent le même répertoire (`cache_dir`).
- **Spécificités** :
  - Force `VLLM_USE_TRUST_REMOTE_CODE=1` pour charger les kernels personnalisés.
  - Manipule `CUDA_VISIBLE_DEVICES` pour limiter la visibilité GPU au sous-ensemble choisi.
  - Les prompts sont générés avec `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
  - Dépend d'un environnement PyTorch compilé pour CUDA 12.4 (`torch==2.6.0+cu124`). Toute divergence apparaît immédiatement dans le tableau `dependencies` du dashboard.
- **Exemple d'appel** :
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen", "messages": [{"role": "user", "content": "Quel est le runtime CUDA installé ?"}]}'
  ```
  La réponse doit mentionner explicitement CUDA 12.4 si le backend est correctement configuré.

## NVIDIA Canary 1B v2 (Reconnaissance vocale)

- **Identifiant Hugging Face** : `nvidia/canary-1b-v2`.
- **Format** : archive `.nemo` (NeMo Toolkit).
- **Pipeline** :
  - `ASRModel.restore_from` restaure les poids et configure le modèle sur `cuda:<id>`.
  - Résolution audio : convertit l'entrée en mono float32 et rééchantillonne à 16 kHz.
  - Utilise un fichier temporaire `.wav` pour passer les données au pipeline NeMo (certains modèles attendent un chemin disque).
- **Langues** : multilingue ; le wrapper sélectionne `source_lang="en"`, `target_lang="en"` (adapter si besoin).
- **Contraintes** :
  - Requiert `nemo_toolkit[asr]==1.23.0` et un GPU CUDA visible.
  - Le fichier `.nemo` (~3.2 Go) doit être présent dans le cache avant le chargement.
- **Bonnes pratiques audio** :
  - Pré-convertir avec `ffmpeg -i input.mp3 -ac 1 -ar 16000 -sample_fmt s16 sample.wav` pour limiter la charge CPU.
  - Surveiller `dependency_inspector` pour vérifier que `torchaudio` est compilé en `+cu124`.

## Pyannote speaker diarization

- **Identifiant Hugging Face** : `pyannote/speaker-diarization-3.1`.
- **Pile logicielle** : `pyannote.audio>=3.4.0`, `torch==2.6.0+cu124`, `torchaudio==2.6.0+cu124`.
- **Téléchargement** :
  - Autorise les motifs `*.bin`, `*.ckpt`, `*.pt`, `*.yaml`, `*.json`.
  - Les références `$MODEL/...` dans les fichiers de config sont résolues en chemins locaux via un patch temporaire du getter Pyannote.
  - **Initialisation** :
    - Vérifie la compatibilité de la version Pyannote (`>=3.4`) et la disponibilité CUDA.
  - Déplace explicitement le pipeline sur `cuda:<id>` choisi.
  - Nettoie les patches (`get_model`) après initialisation pour éviter de modifier l'environnement global.
- **Sortie** :
  - Liste de segments `{ speaker, start, end }` (float secondes).
  - Aucun post-traitement additionnel n'est appliqué (la temporalité brute du pipeline est conservée).
- **Astuce de validation** :
  - L'endpoint `POST /api/diarization/process` renvoie `runtime.details.sample_rate` dans le tableau de bord. Une valeur différente de `16000` peut indiquer un mauvais resampling (souvent dû à une installation torchaudio CPU-only).

## Bonnes pratiques générales

- Toujours s'assurer que le cache (`MODEL_CACHE_DIR`) est partagé entre les redéploiements pour éviter les téléchargements répétitifs.
- Pour forcer un re-téléchargement propre, supprimer le dossier `models--<org>--<repo>` correspondant avant d'appeler l'endpoint `download`.
- Adapter `preferred_device_ids` en fonction du plan de charge GPU (vLLM peut saturer un GPU ; Canary et Pyannote utilisent un seul GPU chacun).
- Surveiller `runtime.last_error` dans le tableau de bord pour diagnostiquer rapidement les échecs d'initialisation.
- L'entrée `torch` dans `dependencies` doit toujours afficher `"cuda_runtime": "12.4"` ; sinon, redéployer l'image ou mettre à jour les pilotes hôtes.

Pour des instructions d'exploitation complètes, voir `doc/operations.md`.
