import CloudDoneRoundedIcon from "@mui/icons-material/CloudDoneRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import ErrorOutlineRoundedIcon from "@mui/icons-material/ErrorOutlineRounded";
import LanRoundedIcon from "@mui/icons-material/LanRounded";
import MemoryRoundedIcon from "@mui/icons-material/MemoryRounded";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import RocketLaunchRoundedIcon from "@mui/icons-material/RocketLaunchRounded";
import SecurityRoundedIcon from "@mui/icons-material/SecurityRounded";
import SensorsRoundedIcon from "@mui/icons-material/SensorsRounded";
import ShieldMoonRoundedIcon from "@mui/icons-material/ShieldMoonRounded";
import SyncRoundedIcon from "@mui/icons-material/SyncRounded";
import TaskAltRoundedIcon from "@mui/icons-material/TaskAltRounded";
import TroubleshootRoundedIcon from "@mui/icons-material/TroubleshootRounded";
import WarningAmberRoundedIcon from "@mui/icons-material/WarningAmberRounded";
import {
  Alert,
  Avatar,
  Box,
  Card,
  CardActions,
  CardContent,
  CardHeader,
  Checkbox,
  Chip,
  Collapse,
  Divider,
  Fade,
  Grid,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Tooltip,
  Typography,
  alpha,
  useTheme
} from "@mui/material";
import { LoadingButton } from "@mui/lab";
import { SelectChangeEvent } from "@mui/material/Select";
import { keyframes } from "@mui/system";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import heroLogo from "../assets/unified-logo.svg";
import {
  HuggingFaceTokenStatus,
  ModelRuntimeInfo,
  downloadModel,
  fetchHuggingFaceTokenStatus,
  loadModel,
  unloadModel,
  updateHuggingFaceToken
} from "../services/api";
import { useDashboard } from "../hooks/useDashboard";

const float = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-6px); }
  100% { transform: translateY(0px); }
`;

const pulse = keyframes`
  0% { box-shadow: 0 0 0 0 rgba(125, 211, 252, 0.35); }
  70% { box-shadow: 0 0 0 18px rgba(125, 211, 252, 0); }
  100% { box-shadow: 0 0 0 0 rgba(125, 211, 252, 0); }
`;

const spinner = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined) {
    return "—";
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(", ") : "—";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
  return String(value);
};

const getStateChip = (runtime?: ModelRuntimeInfo | null) => {
  const state = runtime?.state ?? "idle";
  switch (state) {
    case "ready":
      return {
        icon: <TaskAltRoundedIcon fontSize="small" />,
        color: "success" as const,
        label: "En ligne"
      };
    case "loading":
      return {
        icon: <SyncRoundedIcon fontSize="small" sx={{ animation: `${spinner} 1.1s linear infinite` }} />,
        color: "info" as const,
        label: "Chargement"
      };
    case "error":
      return {
        icon: <ErrorOutlineRoundedIcon fontSize="small" />,
        color: "error" as const,
        label: "Erreur"
      };
    default:
      return {
        icon: <CloudDownloadRoundedIcon fontSize="small" />,
        color: "default" as const,
        label: "Prêt"
      };
  }
};

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
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["dashboard"] })
  });

  const downloadMutation = useMutation({
    mutationFn: downloadModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["dashboard"] })
  });

  const unloadMutation = useMutation({
    mutationFn: unloadModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["dashboard"] })
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
          background: "linear-gradient(135deg, rgba(14,116,144,0.8) 0%, rgba(76,29,149,0.75) 55%, rgba(236,72,153,0.7) 100%)",
          border: `1px solid ${alpha(theme.palette.common.white, 0.18)}`,
          boxShadow: "0 25px 50px -12px rgba(15,118,110,0.55)",
          backdropFilter: "blur(18px)",
          "::after": {
            content: "''",
            position: "absolute",
            inset: -80,
            background: "radial-gradient(circle at top left, rgba(125,211,252,0.35), transparent 55%)",
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
              width: { xs: 96, md: 128 },
              height: { xs: 96, md: 128 },
              animation: `${float} 6s ease-in-out infinite`,
              filter: "drop-shadow(0 12px 25px rgba(125,211,252,0.55))"
            }}
          />
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="overline" sx={{ letterSpacing: 2, opacity: 0.85 }}>
              Centre de contrôle des modèles
            </Typography>
            <Typography variant="h3" sx={{ fontWeight: 700, mt: 1 }}>
              Orchestration visuelle des charges IA
            </Typography>
            <Typography variant="subtitle1" sx={{ mt: 1.5, maxWidth: 640, opacity: 0.85 }}>
              Surveillez la disponibilité, les téléchargements et l'initialisation GPU en temps réel.
              Chaque carte offre désormais un suivi précis du cache local, des serveurs exposés et
              des étapes de chargement.
            </Typography>
            <Stack direction="row" spacing={1.5} sx={{ mt: 3, flexWrap: "wrap" }}>
              <Chip
                icon={<RocketLaunchRoundedIcon />}
                label={heroSubtitle}
                sx={{
                  bgcolor: alpha(theme.palette.common.black, 0.35),
                  color: theme.palette.common.white,
                  borderColor: alpha(theme.palette.common.white, 0.25),
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
                  borderColor: alpha(theme.palette.common.white, 0.25),
                  borderWidth: 1,
                  borderStyle: "solid"
                }}
              />
            </Stack>
          </Box>
        </Stack>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={5} lg={4}>
          <Card
            sx={{
              height: "100%",
              borderRadius: 3,
              background: alpha(theme.palette.background.paper, 0.9),
              border: `1px solid ${alpha(theme.palette.primary.main, 0.25)}`,
              boxShadow: "0 12px 40px rgba(30,64,175,0.25)"
            }}
          >
            <CardHeader
              avatar={<Avatar sx={{ bgcolor: theme.palette.primary.main }}><SecurityRoundedIcon /></Avatar>}
              title="Jeton Hugging Face"
              subheader="Gestion centralisée de l'authentification"
            />
            <Divider sx={{ opacity: 0.2 }} />
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Fournissez un jeton personnel pour permettre le téléchargement automatisé des modèles
                privés directement sur le serveur GPU.
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
              <Stack
                direction={{ xs: "column", sm: "row" }}
                spacing={2}
                sx={{ mt: 2 }}
              >
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

        <Grid item xs={12} md={7} lg={8}>
          <Grid container spacing={2}>
            {data?.gpus?.map((gpu) => {
              const memoryPercent = gpu.memory_total
                ? Math.round((gpu.memory_used / gpu.memory_total) * 100)
                : 0;
              return (
                <Grid item xs={12} md={6} key={gpu.id}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 2.5,
                      borderRadius: 3,
                      border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                      background: alpha(theme.palette.background.default, 0.75),
                      backdropFilter: "blur(16px)",
                      position: "relative"
                    }}
                  >
                    <Stack direction="row" spacing={2} alignItems="center">
                      <Avatar sx={{ bgcolor: alpha(theme.palette.primary.light, 0.25), color: theme.palette.primary.light }}>
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
                  </Paper>
                </Grid>
              );
            })}
          </Grid>
        </Grid>

        <Grid item xs={12}>
          {(isLoading ||
            loadMutation.isPending ||
            unloadMutation.isPending ||
            tokenMutation.isPending ||
            tokenQuery.isLoading) && <LinearProgress sx={{ mb: 2 }} />}
        </Grid>

        {data &&
          Object.entries(data.models).map(([key, model]) => {
            const runtime = model.runtime;
            const gpuOptions = data.gpus ?? [];
            const activeDeviceIds = Array.isArray(model.params?.device_ids)
              ? (model.params?.device_ids as number[]).map((value) => Number(value))
              : [];
            const selectedForModel = selectedDevices[key] ?? activeDeviceIds;
            const progress = runtime?.progress ?? (model.loaded ? 100 : 0);
            const stateChip = getStateChip(runtime);
            const downloaded = runtime?.downloaded ?? false;
            const serverEntries = runtime?.server ? Object.entries(runtime.server) : [];
            const detailEntries = runtime?.details ? Object.entries(runtime.details) : [];
            const isLoadingAction = loadMutation.isPending && loadMutation.variables?.key === key;
            const isDownloadingAction = downloadMutation.isPending && downloadMutation.variables === key;
            const isUnloadingAction = unloadMutation.isPending && unloadMutation.variables === key;
            const statusMessage = runtime?.status ?? (model.loaded ? "Modèle prêt" : "Pas encore chargé");

            const handleDeviceChange = (event: SelectChangeEvent<string[]>) => {
              const value = event.target.value;
              const normalized = (typeof value === "string" ? value.split(",") : value)
                .filter((item) => item !== "")
                .map((item) => Number(item));
              setSelectedDevices((prev) => ({ ...prev, [key]: normalized }));
            };

            return (
              <Grid item xs={12} md={4} key={key}>
                <Card
                  sx={{
                    height: "100%",
                    borderRadius: 3,
                    position: "relative",
                    overflow: "hidden",
                    border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                    background: alpha(theme.palette.background.paper, 0.85),
                    backdropFilter: "blur(16px)",
                    animation: runtime?.state === "loading" ? `${pulse} 2.8s ease-in-out infinite` : "none"
                  }}
                >
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      opacity: 0.18,
                      background: `radial-gradient(circle at top, ${theme.palette.primary.main}, transparent 60%)`
                    }}
                  />
                  <CardHeader
                    avatar={<Avatar sx={{ bgcolor: alpha(theme.palette.primary.main, 0.6) }}>{model.identifier[0]}</Avatar>}
                    title={model.identifier}
                    subheader={model.description}
                  />
                  <CardContent sx={{ position: "relative", zIndex: 1 }}>
                    <Stack direction="row" spacing={1.5} sx={{ flexWrap: "wrap" }}>
                      <Chip icon={stateChip.icon} color={stateChip.color} label={stateChip.label} />
                      <Chip
                        icon={downloaded ? <CloudDoneRoundedIcon /> : <CloudDownloadRoundedIcon />}
                        color={downloaded ? "success" : "default"}
                        label={downloaded ? "En cache" : "Téléchargement requis"}
                      />
                      <Chip
                        icon={<TroubleshootRoundedIcon fontSize="small" />}
                        label={`Format: ${model.format}`}
                      />
                    </Stack>
                    <Box sx={{ mt: 2 }}>
                      <Stack direction="row" alignItems="center" spacing={2}>
                        <Typography variant="body2" sx={{ minWidth: 72 }}>
                          Progression
                        </Typography>
                        <Tooltip title={`${progress}%`}>
                          <Box sx={{ flexGrow: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={progress}
                              sx={{
                                height: 10,
                                borderRadius: 6,
                                backgroundColor: alpha(theme.palette.common.white, 0.08),
                                "& .MuiLinearProgress-bar": {
                                  borderRadius: 6,
                                  background: `linear-gradient(90deg, ${theme.palette.primary.light}, ${theme.palette.secondary.main})`,
                                  transition: "transform 0.5s ease"
                                }
                              }}
                            />
                          </Box>
                        </Tooltip>
                        <Typography variant="body2" sx={{ width: 40 }}>
                          {Math.round(progress)}%
                        </Typography>
                      </Stack>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
                        {statusMessage}
                      </Typography>
                      {runtime?.updated_at && (
                        <Typography variant="caption" color="text.secondary">
                          Dernière mise à jour · {new Date(runtime.updated_at).toLocaleTimeString()}
                        </Typography>
                      )}
                    </Box>

                    {detailEntries.length > 0 && (
                      <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap" }}>
                        {detailEntries.map(([detailKey, detailValue]) => (
                          <Chip
                            key={detailKey}
                            icon={<SensorsRoundedIcon fontSize="small" />}
                            label={`${detailKey}: ${formatValue(detailValue)}`}
                            size="small"
                            sx={{ bgcolor: alpha(theme.palette.primary.light, 0.12) }}
                          />
                        ))}
                      </Stack>
                    )}

                    <Collapse in={serverEntries.length > 0} timeout={300} unmountOnExit>
                      <Paper
                        variant="outlined"
                        sx={{
                          mt: 2,
                          p: 2,
                          borderRadius: 2.5,
                          borderColor: alpha(theme.palette.primary.main, 0.25)
                        }}
                      >
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                          <LanRoundedIcon fontSize="small" />
                          <Typography variant="subtitle2">Point d'accès serveur</Typography>
                        </Stack>
                        <Divider sx={{ mb: 1, opacity: 0.4 }} />
                        <List dense>
                          {serverEntries.map(([serverKey, serverValue]) => (
                            <ListItem key={serverKey} disablePadding>
                              <ListItemIcon sx={{ minWidth: 32 }}>
                                <PlayArrowRoundedIcon fontSize="small" />
                              </ListItemIcon>
                              <ListItemText
                                primary={serverKey}
                                secondary={formatValue(serverValue)}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Paper>
                    </Collapse>

                    {runtime?.state === "error" && runtime.last_error && (
                      <Fade in>
                        <Alert severity="error" sx={{ mt: 2 }}>
                          {runtime.last_error}
                        </Alert>
                      </Fade>
                    )}
                  </CardContent>
                  <Divider sx={{ opacity: 0.15 }} />
                  <CardActions sx={{ px: 3, pb: 3 }}>
                    <LoadingButton
                      size="small"
                      variant="outlined"
                      onClick={() => downloadMutation.mutate(key)}
                      disabled={downloaded || runtime?.state === "loading" || isLoadingAction}
                      loading={isDownloadingAction}
                      startIcon={<CloudDownloadRoundedIcon />}
                    >
                      Télécharger
                    </LoadingButton>
                    <LoadingButton
                      size="small"
                      variant="contained"
                      onClick={() => loadMutation.mutate({ key, gpuDeviceIds: selectedDevices[key] })}
                      disabled={model.loaded || runtime?.state === "loading" || isDownloadingAction}
                      loading={isLoadingAction}
                      startIcon={<PlayArrowRoundedIcon />}
                    >
                      Charger
                    </LoadingButton>
                    <LoadingButton
                      size="small"
                      variant="outlined"
                      color="warning"
                      onClick={() => unloadMutation.mutate(key)}
                      disabled={!model.loaded || runtime?.state === "loading" || isDownloadingAction}
                      loading={isUnloadingAction}
                      startIcon={<WarningAmberRoundedIcon />}
                    >
                      Décharger
                    </LoadingButton>
                    <Select
                      multiple
                      displayEmpty
                      size="small"
                      value={selectedForModel.map((item) => String(item))}
                      onChange={handleDeviceChange}
                      disabled={gpuOptions.length === 0}
                      sx={{
                        ml: "auto",
                        minWidth: 140,
                        backgroundColor: alpha(theme.palette.background.paper, 0.6),
                        borderRadius: 2
                      }}
                      MenuProps={{ disableCloseOnSelect: true }}
                      renderValue={(selected) =>
                        selected.length === 0
                          ? "GPU auto"
                          : (selected as string[])
                              .map((item) => Number(item))
                              .join(", ")
                      }
                    >
                      {gpuOptions.map((gpu) => (
                        <MenuItem
                          key={gpu.id}
                          value={String(gpu.id)}
                        >
                          <Checkbox checked={selectedForModel.includes(gpu.id)} />
                          <ListItemText primary={`GPU ${gpu.id} (${gpu.name})`} />
                        </MenuItem>
                      ))}
                    </Select>
                  </CardActions>
                </Card>
              </Grid>
            );
          })}
      </Grid>
    </Box>
  );
};

export default ModelsPage;
