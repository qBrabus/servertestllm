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

## Pyannote speaker diarization

- **Identifiant Hugging Face** : `pyannote/speaker-diarization-community-1`.
- **Pile logicielle** : `pyannote.audio>=4.0.1`, `torch==2.8.0`.
- **Téléchargement** :
  - Autorise les motifs `*.bin`, `*.ckpt`, `*.pt`, `*.yaml`, `*.json`.
  - Les références `$MODEL/...` dans les fichiers de config sont résolues en chemins locaux via un patch temporaire du getter Pyannote.
- **Initialisation** :
  - Vérifie la compatibilité de la version Pyannote (`>=4`) et la disponibilité CUDA.
  - Déplace explicitement le pipeline sur `cuda:<id>` choisi.
  - Nettoie les patches (`get_model`) après initialisation pour éviter de modifier l'environnement global.
- **Sortie** :
  - Liste de segments `{ speaker, start, end }` (float secondes).
  - Aucun post-traitement additionnel n'est appliqué (la temporalité brute du pipeline est conservée).

## Bonnes pratiques générales

- Toujours s'assurer que le cache (`MODEL_CACHE_DIR`) est partagé entre les redéploiements pour éviter les téléchargements répétitifs.
- Pour forcer un re-téléchargement propre, supprimer le dossier `models--<org>--<repo>` correspondant avant d'appeler l'endpoint `download`.
- Adapter `preferred_device_ids` en fonction du plan de charge GPU (vLLM peut saturer un GPU ; Canary et Pyannote utilisent un seul GPU chacun).
- Surveiller `runtime.last_error` dans le tableau de bord pour diagnostiquer rapidement les échecs d'initialisation.

Pour des instructions d'exploitation complètes, voir `doc/operations.md`.
