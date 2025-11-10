import AutoAwesomeRoundedIcon from "@mui/icons-material/AutoAwesomeRounded";
import CloudDoneRoundedIcon from "@mui/icons-material/CloudDoneRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import MemoryRoundedIcon from "@mui/icons-material/MemoryRounded";
import RocketLaunchRoundedIcon from "@mui/icons-material/RocketLaunchRounded";
import SensorsRoundedIcon from "@mui/icons-material/SensorsRounded";
import ShieldMoonRoundedIcon from "@mui/icons-material/ShieldMoonRounded";
import SpeedRoundedIcon from "@mui/icons-material/SpeedRounded";
import WarningAmberRoundedIcon from "@mui/icons-material/WarningAmberRounded";
import {
  Alert,
  Avatar,
  Box,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Divider,
  Grid,
  LinearProgress,
  Stack,
  TextField,
  Typography,
  alpha,
  useTheme
} from "@mui/material";
import { LoadingButton } from "@mui/lab";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import ModelCard from "../components/ModelCard";
import heroLogo from "../assets/unified-logo.svg";
import {
  DashboardState,
  HuggingFaceTokenStatus,
  ModelInfo,
  ModelRuntimeInfo,
  downloadModel,
  fetchHuggingFaceTokenStatus,
  loadModel,
  unloadModel,
  updateHuggingFaceToken
} from "../services/api";
import { useDashboard } from "../hooks/useDashboard";

const ModelsPage = () => {
  const theme = useTheme();
  const queryClient = useQueryClient();
  const { data, isLoading } = useDashboard();
  const [selectedDevices, setSelectedDevices] = useState<Record<string, number[]>>({});
  const [tokenInput, setTokenInput] = useState("");
  const [tokenFeedback, setTokenFeedback] = useState<
    { message: string; severity: "success" | "error" } | null
  >(null);

  const totalModels = data ? Object.keys(data.models).length : 0;
  const loadedModels = data
    ? Object.values(data.models).filter((model) => model.loaded).length
    : 0;
  const cachedModels = data
    ? Object.values(data.models).filter((model) => model.runtime?.downloaded).length
    : 0;

  const tokenQuery = useQuery<HuggingFaceTokenStatus>({
    queryKey: ["huggingface-token"],
    queryFn: fetchHuggingFaceTokenStatus,
    refetchInterval: 60000
  });

  const loadMutation = useMutation({
    mutationFn: loadModel,
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["dashboard"] })
  });

  type DownloadMutationContext = { previousState?: DashboardState };

  const downloadMutation = useMutation<Record<string, ModelInfo>, Error, string, DownloadMutationContext>({
    mutationFn: downloadModel,
    onMutate: async (key) => {
      const previousState = queryClient.getQueryData<DashboardState>(["dashboard"]);
      if (previousState && previousState.models[key]) {
        const currentModel = previousState.models[key];
        const previousRuntime = currentModel.runtime ?? null;
        const optimisticRuntime: ModelRuntimeInfo = {
          state: "loading",
          progress: Math.max(previousRuntime?.progress ?? 0, 8),
          status: "Préparation du téléchargement...",
          details: previousRuntime?.details ?? null,
          server: previousRuntime?.server ?? null,
          downloaded: false,
          last_error: null,
          updated_at: new Date().toISOString()
        };

        const nextState: DashboardState = {
          ...previousState,
          models: {
            ...previousState.models,
            [key]: {
              ...currentModel,
              runtime: optimisticRuntime
            }
          }
        };

        queryClient.setQueryData(["dashboard"], nextState);
      }

      void queryClient.invalidateQueries({ queryKey: ["dashboard"], refetchType: "active" });

      return { previousState };
    },
    onError: (_error, _key, context) => {
      if (context?.previousState) {
        queryClient.setQueryData(["dashboard"], context.previousState);
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["dashboard"] });
    }
  });

  const unloadMutation = useMutation({
    mutationFn: unloadModel,
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["dashboard"] })
  });

  const tokenMutation = useMutation<HuggingFaceTokenStatus, Error, string | null>({
    mutationFn: updateHuggingFaceToken,
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["huggingface-token"] });
      setTokenInput("");
      setTokenFeedback({
        message: result.has_token
          ? "Jeton Hugging Face enregistré avec succès."
          : "Jeton supprimé du serveur.",
        severity: "success"
      });
    },
    onError: (error) => {
      setTokenFeedback({
        message: error?.message ?? "Impossible de mettre à jour le jeton.",
        severity: "error"
      });
    }
  });

  const heroSubtitle = useMemo(() => {
    if (!totalModels) {
      return "Préparez vos modèles pour l'orchestration hybride.";
    }
    return `${loadedModels}/${totalModels} modèles en ligne · ${cachedModels} déjà en cache local`;
  }, [totalModels, loadedModels, cachedModels]);

  const systemMetrics = data?.system;

  const isBusy =
    isLoading ||
    loadMutation.isPending ||
    unloadMutation.isPending ||
    tokenMutation.isPending ||
    downloadMutation.isPending ||
    tokenQuery.isLoading;

  return (
    <Box sx={{ position: "relative", zIndex: 1 }}>
      <Box
        sx={{
          position: "relative",
          mb: 4,
          p: { xs: 3, md: 5 },
          borderRadius: 4,
          overflow: "hidden",
          color: theme.palette.common.white,
          background: "linear-gradient(135deg, rgba(15,118,110,0.82) 0%, rgba(79,70,229,0.82) 55%, rgba(236,72,153,0.75) 100%)",
          border: `1px solid ${alpha(theme.palette.common.white, 0.18)}`,
          boxShadow: "0 30px 60px -25px rgba(15,118,110,0.55)",
          backdropFilter: "blur(20px)",
          "::after": {
            content: "''",
            position: "absolute",
            inset: -120,
            background: "radial-gradient(circle at top left, rgba(125,211,252,0.35), transparent 60%)",
            pointerEvents: "none"
          }
        }}
      >
        <Stack direction={{ xs: "column", md: "row" }} spacing={4} alignItems="center">
          <Box
            component="img"
            src={heroLogo}
            alt="Unified Gateway"
            sx={{
              width: { xs: 108, md: 140 },
              height: { xs: 108, md: 140 },
              filter: "drop-shadow(0 18px 40px rgba(125,211,252,0.55))",
              animation: "floatHero 8s ease-in-out infinite"
            }}
          />
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="overline" sx={{ letterSpacing: 2, opacity: 0.85 }}>
              Unified Inference Control Plane
            </Typography>
            <Typography variant="h3" sx={{ fontWeight: 700, mt: 1 }}>
              Pilotage visuel des pipelines IA & audio
            </Typography>
            <Typography variant="subtitle1" sx={{ mt: 1.5, maxWidth: 640, opacity: 0.9 }}>
              Surveillez la disponibilité, la mise en cache et les points d’accès API de vos modèles GPU.
              Chaque carte propose désormais une télémétrie en direct, des actions rapides et la copie immédiate des
              endpoints exposés.
            </Typography>
            <Stack direction="row" spacing={1.5} sx={{ mt: 3, flexWrap: "wrap" }}>
              <Chip
                icon={<RocketLaunchRoundedIcon />}
                label={heroSubtitle}
                sx={{
                  bgcolor: alpha(theme.palette.common.black, 0.35),
                  color: theme.palette.common.white,
                  borderColor: alpha(theme.palette.common.white, 0.3),
                  borderWidth: 1,
                  borderStyle: "solid"
                }}
              />
              <Chip
                icon={<SensorsRoundedIcon />}
                label={`${data?.gpus?.length ?? 0} GPU détecté(s)`}
                sx={{
                  bgcolor: alpha(theme.palette.common.black, 0.35),
                  color: theme.palette.common.white,
                  borderColor: alpha(theme.palette.common.white, 0.3),
                  borderWidth: 1,
                  borderStyle: "solid"
                }}
              />
            </Stack>
          </Box>
        </Stack>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card
            sx={{
              height: "100%",
              borderRadius: 4,
              border: `1px solid ${alpha(theme.palette.primary.main, 0.25)}`,
              background: alpha(theme.palette.background.paper, 0.9),
              boxShadow: "0 18px 45px rgba(30,64,175,0.25)"
            }}
          >
            <CardHeader
              avatar={<Avatar sx={{ bgcolor: theme.palette.primary.main }}><CloudDownloadRoundedIcon /></Avatar>}
              title="Statut des modèles"
              subheader="Vue agrégée de l'orchestration"
            />
            <Divider sx={{ opacity: 0.2 }} />
            <CardContent>
              <Stack spacing={1.5}>
                <Stack direction="row" spacing={2} alignItems="center">
                  <Avatar sx={{ bgcolor: alpha(theme.palette.success.main, 0.15), color: theme.palette.success.main }}>
                    <CloudDoneRoundedIcon />
                  </Avatar>
                  <Box>
                    <Typography variant="subtitle1">Modèles prêts</Typography>
                    <Typography variant="h5" sx={{ fontWeight: 700 }}>
                      {loadedModels}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {totalModels > 0 ? `${loadedModels}/${totalModels} actifs` : "Aucun modèle enregistré"}
                    </Typography>
                  </Box>
                </Stack>
                <Divider flexItem sx={{ opacity: 0.1 }} />
                <Stack direction="row" spacing={2} alignItems="center">
                  <Avatar sx={{ bgcolor: alpha(theme.palette.info.main, 0.15), color: theme.palette.info.main }}>
                    <AutoAwesomeRoundedIcon />
                  </Avatar>
                  <Box>
                    <Typography variant="subtitle1">Artefacts en cache</Typography>
                    <Typography variant="h5" sx={{ fontWeight: 700 }}>
                      {cachedModels}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Téléchargements terminés et prêts à être chargés
                    </Typography>
                  </Box>
                </Stack>
                <Divider flexItem sx={{ opacity: 0.1 }} />
                <Stack direction="row" spacing={2} alignItems="center">
                  <Avatar sx={{ bgcolor: alpha(theme.palette.secondary.main, 0.15), color: theme.palette.secondary.main }}>
                    <SpeedRoundedIcon />
                  </Avatar>
                  <Box>
                    <Typography variant="subtitle1">Santé du serveur</Typography>
                    <Typography variant="body2" color="text.secondary">
                      CPU {systemMetrics ? systemMetrics.cpu_percent.toFixed(1) : "—"}% · Mémoire {systemMetrics ? systemMetrics.memory_percent.toFixed(1) : "—"}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={systemMetrics ? systemMetrics.cpu_percent : 0}
                      sx={{ mt: 1, height: 8, borderRadius: 4 }}
                    />
                  </Box>
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card
            sx={{
              height: "100%",
              borderRadius: 4,
              border: `1px solid ${alpha(theme.palette.primary.main, 0.25)}`,
              background: alpha(theme.palette.background.paper, 0.9),
              boxShadow: "0 18px 45px rgba(30,64,175,0.25)"
            }}
          >
            <CardHeader
              avatar={<Avatar sx={{ bgcolor: theme.palette.primary.main }}><ShieldMoonRoundedIcon /></Avatar>}
              title="Jeton Hugging Face"
              subheader="Gestion centralisée"
            />
            <Divider sx={{ opacity: 0.2 }} />
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Fournissez un jeton personnel pour permettre le téléchargement automatisé des modèles privés directement sur le serveur GPU.
              </Typography>
              <TextField
                type="password"
                label="Jeton"
                fullWidth
                sx={{ mt: 2 }}
                value={tokenInput}
                onChange={(event) => {
                  setTokenInput(event.target.value);
                  if (tokenFeedback) {
                    setTokenFeedback(null);
                  }
                }}
                placeholder={
                  tokenQuery.data?.has_token
                    ? "Un jeton est déjà enregistré"
                    : "Coller ici votre jeton Hugging Face"
                }
                helperText={
                  tokenQuery.data?.has_token
                    ? "Un jeton est stocké. Soumettre un nouveau le remplacera."
                    : "Aucun jeton enregistré pour l'instant."
                }
              />
              <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mt: 2 }}>
                <LoadingButton
                  variant="contained"
                  onClick={() => tokenMutation.mutate(tokenInput.trim())}
                  disabled={tokenInput.trim().length === 0}
                  loading={tokenMutation.isPending}
                  startIcon={<ShieldMoonRoundedIcon />}
                >
                  Enregistrer
                </LoadingButton>
                <LoadingButton
                  variant="outlined"
                  color="warning"
                  onClick={() => tokenMutation.mutate(null)}
                  disabled={!tokenQuery.data?.has_token}
                  loading={tokenMutation.isPending}
                  startIcon={<WarningAmberRoundedIcon />}
                >
                  Réinitialiser
                </LoadingButton>
              </Stack>
              {tokenFeedback && (
                <Alert
                  severity={tokenFeedback.severity}
                  sx={{ mt: 2 }}
                  onClose={() => setTokenFeedback(null)}
                >
                  {tokenFeedback.message}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card
            sx={{
              height: "100%",
              borderRadius: 4,
              border: `1px solid ${alpha(theme.palette.primary.main, 0.25)}`,
              background: alpha(theme.palette.background.paper, 0.9),
              boxShadow: "0 18px 45px rgba(30,64,175,0.25)"
            }}
          >
            <CardHeader
              avatar={<Avatar sx={{ bgcolor: theme.palette.primary.main }}><MemoryRoundedIcon /></Avatar>}
              title="GPU Monitor"
              subheader="Utilisation en temps réel"
            />
            <Divider sx={{ opacity: 0.2 }} />
            <CardContent>
              <Stack spacing={2}>
                {data?.gpus?.map((gpu) => {
                  const memoryPercent = gpu.memory_total
                    ? Math.round((gpu.memory_used / gpu.memory_total) * 100)
                    : 0;
                  return (
                    <Box
                      key={gpu.id}
                      sx={{
                        p: 2,
                        borderRadius: 3,
                        border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                        background: alpha(theme.palette.background.default, 0.65),
                        backdropFilter: "blur(18px)"
                      }}
                    >
                      <Stack direction="row" spacing={2} alignItems="center">
                        <Avatar sx={{ bgcolor: alpha(theme.palette.primary.light, 0.15), color: theme.palette.primary.light }}>
                          <MemoryRoundedIcon />
                        </Avatar>
                        <Box sx={{ flexGrow: 1 }}>
                          <Typography variant="subtitle1">GPU {gpu.id} · {gpu.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            Charge: {Math.round(gpu.load * 100)}% · Température: {gpu.temperature ?? "n/a"}°C
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={memoryPercent}
                            sx={{
                              mt: 1,
                              height: 8,
                              borderRadius: 4,
                              backgroundColor: alpha(theme.palette.common.white, 0.08),
                              "& .MuiLinearProgress-bar": {
                                borderRadius: 4,
                                background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`
                              }
                            }}
                          />
                          <Typography variant="caption" color="text.secondary">
                            {gpu.memory_used.toFixed(1)} / {gpu.memory_total.toFixed(1)} Go utilisés
                          </Typography>
                        </Box>
                      </Stack>
                    </Box>
                  );
                })}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          {isBusy && <LinearProgress sx={{ mb: 2 }} />}
        </Grid>

        {data && (
          <Grid item xs={12}>
            <Grid container spacing={3}>
              {Object.entries(data.models).map(([key, model]) => {
                const runtime = model.runtime;
                const gpuOptions = data.gpus ?? [];
                const activeDeviceIds = Array.isArray(model.params?.device_ids)
                  ? (model.params?.device_ids as number[]).map((value) => Number(value))
                  : [];
                const selectedForModel = selectedDevices[key] ?? activeDeviceIds;
                const isLoadingAction = loadMutation.isPending && loadMutation.variables?.key === key;
                const isDownloadingAction = downloadMutation.isPending && downloadMutation.variables === key;
                const isUnloadingAction = unloadMutation.isPending && unloadMutation.variables === key;

                return (
                  <Grid item xs={12} md={6} xl={4} key={key}>
                    <ModelCard
                      modelKey={key}
                      model={model}
                      runtime={runtime}
                      gpuOptions={gpuOptions}
                      selectedDevices={selectedForModel}
                      onDeviceChange={(devices) => setSelectedDevices((prev) => ({ ...prev, [key]: devices }))}
                      onDownload={() => downloadMutation.mutate(key)}
                      onLoad={() => loadMutation.mutate({ key, gpuDeviceIds: selectedDevices[key] })}
                      onUnload={() => unloadMutation.mutate(key)}
                      isDownloading={isDownloadingAction}
                      isLoading={isLoadingAction}
                      isUnloading={isUnloadingAction}
                    />
                  </Grid>
                );
              })}
            </Grid>
          </Grid>
        )}
      </Grid>

      <style>{`
        @keyframes floatHero {
          0% { transform: translateY(0px); }
          50% { transform: translateY(-6px); }
          100% { transform: translateY(0px); }
        }
      `}</style>
    </Box>
  );
};

export default ModelsPage;
