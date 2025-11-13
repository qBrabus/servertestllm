# Frontend React

Le tableau de bord d'administration est une application **React 18** construite avec **Vite**, **TypeScript** et **Material UI**. Son objectif : présenter en temps réel l'état des modèles, offrir des actions de pilotage (télécharger/charger/décharger) et fournir des formulaires de test (chat, audio).

## Structure des dossiers (`frontend/src`)

- `App.tsx` : composition principale (AppBar, navigation latérale, routes).
- `pages/` :
  - `DashboardPage.tsx` : vue synthétique (état GPU, progression des modèles, dépendances).
  - `ModelsPage.tsx` : détail et actions par modèle.
  - `AudioPage.tsx` : formulaires d'upload pour transcription et diarisation.
  - `PlaygroundPage.tsx` : interface minimaliste pour l'API chat/completions.
- `components/` :
  - `ApiKeyDialog.tsx` : gestion de la clé API (stockage local + dialogue Material UI).
  - `ModelCard.tsx`, `TorchStackCard.tsx` : cartes visuelles pour modèles/dépendances.
- `hooks/` :
  - `useDashboard.ts` : encapsule la requête `GET /api/admin/status` via React Query.
- `services/api.ts` : client Axios centralisé, définitions TypeScript des réponses et helpers pour chaque endpoint.
- `assets/` : logo SVG affiché dans la barre supérieure.

## Flux de données

1. `App.tsx` configure le layout (drawer responsive) et enregistre les routes avec `react-router-dom`.
2. Les pages consomment les hooks/services (principalement `useDashboard` et les fonctions d'API) pour récupérer les données JSON.
3. Les actions (boutons télécharger/charger/décharger) déclenchent des POST via `services/api.ts`; la mutation réussie rafraîchit ensuite le cache React Query.
4. Les barres de progression affichent `runtime.progress` tandis que les badges s'appuient sur `runtime.downloaded` et `runtime.state`.
5. `TorchStackCard` présente l'état CUDA 12.4 en lisant l'entrée `torch` dans `dependencies`; si `cuda` vaut `false` ou que `details.cuda_runtime` diffère, un badge rouge « CPU-only » s'affiche.

## Gestion de la clé API

- Stockée dans `localStorage` quand disponible (`getStoredApiKey`, `setStoredApiKey`).
- Un intercepteur Axios injecte `Authorization: Bearer <clé>` si présent.
- En mode navigateur restreint (Safari privé, Firefox strict), l'absence de `localStorage` est détectée et loggée sans bloquer l'affichage.

## Styles et expérience utilisateur

- Material UI (thème par défaut) enrichi par des gradients et ombres (voir `App.tsx`).
- Drawer permanent en desktop (`sm` et plus), drawer temporaire sur mobile.
- Requêtes à intervalle dynamique : `useDashboard` réduit l'intervalle à 1 s lorsqu'un modèle est en train de charger/télécharger (`runtime.state === "loading"`).
- Les cartes modèles affichent également la VRAM disponible (`gpu.memory.total`) afin de rappeler les contraintes CUDA 12.4 lors du chargement de Qwen.

## Interaction avec le backend

| Action UI | Endpoint backend | Détails |
|-----------|------------------|---------|
| Afficher le statut | `GET /api/admin/status` | Retourne `DashboardState` (GPU, modèles, dépendances, métriques système). |
| Télécharger un modèle | `POST /api/admin/models/{clé}/download` | Lance `ModelRegistry.ensure_downloaded`. |
| Charger un modèle | `POST /api/admin/models/{clé}/load` | Optionnellement avec `gpu_device_ids`. |
| Décharger un modèle | `POST /api/admin/models/{clé}/unload` | Libère la VRAM via `ModelWrapper.unload`. |
| Mettre à jour le token HF | `POST /api/admin/huggingface/token` | Persiste via `TokenStore` et met à jour le registre. |
| Tester le chat | `POST /v1/chat/completions` | Utilise le modèle Qwen via vLLM. |
| Transcrire un audio | `POST /api/audio/transcribe` | Upload `multipart/form-data`. |
| Lancer la diarisation | `POST /api/diarization/process` | Upload `multipart/form-data`. |

## Développement local du frontend

1. Installer les dépendances (`npm install` ou `pnpm install`) dans `frontend/`.
2. Lancer le serveur de dev : `npm run dev -- --host` (permet l'accès depuis le réseau).
3. Configurer le proxy `vite.config.ts` pour rediriger `/api` et `/v1` vers le backend (si lancé séparément).
4. Compiler pour la production : `npm run build`, puis servir `frontend/dist` via le backend (paramètre `FRONTEND_DIST`).
5. Pour simuler les statuts CUDA lors du développement, vous pouvez forcer l'entrée `torch` dans `fetchDashboard` lorsque `import.meta.env.DEV` vaut `true` :
   ```ts
   if (import.meta.env.DEV) {
     data.dependencies = data.dependencies.map((dependency) =>
       dependency.name === "torch"
         ? {
             ...dependency,
             version: "2.6.0+cu124",
             cuda: true,
             details: {
               ...(dependency.details ?? {}),
               cuda_runtime: "12.4",
               cuda_available: true
             }
           }
         : dependency
     );
   }
   ```
   Retirez ce bloc avant commit pour éviter de masquer un problème réel.

## Tests et qualité

- Les composants sont conçus pour être testés avec `@testing-library/react` (non inclus par défaut, à ajouter si nécessaire).
- Pour vérifier la cohérence TypeScript, exécuter `npm run lint` et `npm run typecheck`.
- Les requêtes longues (téléchargements HF) désactivent le timeout Axios (`timeout: 0`) pour éviter les erreurs réseau.
- `services/api.ts` consigne un warning si `dependencies.torch.details.cuda_runtime` ne vaut pas `12.4`, afin de guider le diagnostic côté frontend.

Pour la coordination globale avec le backend et la conteneurisation, consultez `doc/architecture.md` et `doc/operations.md`.
