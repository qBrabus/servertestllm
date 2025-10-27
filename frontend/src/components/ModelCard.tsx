import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import CloudDoneRoundedIcon from "@mui/icons-material/CloudDoneRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import LanRoundedIcon from "@mui/icons-material/LanRounded";
import MemoryRoundedIcon from "@mui/icons-material/MemoryRounded";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import PowerSettingsNewRoundedIcon from "@mui/icons-material/PowerSettingsNewRounded";
import RouterRoundedIcon from "@mui/icons-material/RouterRounded";
import StorageRoundedIcon from "@mui/icons-material/StorageRounded";
import {
  Alert,
  Avatar,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Collapse,
  Divider,
  IconButton,
  LinearProgress,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  MenuItem,
  Select,
  SelectChangeEvent,
  Stack,
  Tooltip,
  Typography,
  alpha,
  useTheme
} from "@mui/material";
import { useMemo, useState } from "react";

import {
  GPUInfo,
  ModelInfo,
  ModelRuntimeInfo
} from "../services/api";

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

const stateLabels: Record<ModelRuntimeInfo["state"], { label: string; color: "default" | "info" | "success" | "error" }> = {
  idle: { label: "Veille", color: "default" },
  loading: { label: "Chargement", color: "info" },
  ready: { label: "En ligne", color: "success" },
  error: { label: "Erreur", color: "error" }
};

export type ModelCardProps = {
  modelKey: string;
  model: ModelInfo;
  runtime?: ModelRuntimeInfo | null;
  gpuOptions: GPUInfo[];
  selectedDevices: number[];
  onDeviceChange: (devices: number[]) => void;
  onDownload: () => void;
  onLoad: () => void;
  onUnload: () => void;
  isDownloading: boolean;
  isLoading: boolean;
  isUnloading: boolean;
};

const ModelCard = ({
  modelKey,
  model,
  runtime,
  gpuOptions,
  selectedDevices,
  onDeviceChange,
  onDownload,
  onLoad,
  onUnload,
  isDownloading,
  isLoading,
  isUnloading
}: ModelCardProps) => {
  const theme = useTheme();
  const [copiedField, setCopiedField] = useState<string | null>(null);

  const state = runtime?.state ?? "idle";
  const progress = runtime?.progress ?? (isDownloading ? 5 : model.loaded ? 100 : 0);
  const downloaded = runtime?.downloaded ?? false;
  const statusMessage = runtime?.status ?? (model.loaded ? "Modèle prêt" : "En attente de chargement");

  const serverEntries = useMemo(() => {
    if (!runtime?.server) {
      return [];
    }
    return Object.entries(runtime.server);
  }, [runtime?.server]);

  const detailEntries = useMemo(() => {
    if (!runtime?.details) {
      return [];
    }
    return Object.entries(runtime.details);
  }, [runtime?.details]);

  const handleDeviceChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    const normalized = (typeof value === "string" ? value.split(",") : value)
      .filter(Boolean)
      .map((item) => Number(item));
    onDeviceChange(normalized);
  };

  const handleCopy = async (label: string, value: string | number | unknown) => {
    if (typeof value === "object") {
      return;
    }
    try {
      await navigator.clipboard?.writeText(String(value));
      setCopiedField(label);
      setTimeout(() => setCopiedField(null), 1500);
    } catch (error) {
      console.warn("Clipboard unavailable", error);
    }
  };

  const stateDescriptor = stateLabels[state] ?? stateLabels.idle;

  return (
    <Card
      key={modelKey}
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        position: "relative",
        overflow: "hidden",
        borderRadius: 4,
        border: `1px solid ${alpha(theme.palette.primary.main, 0.25)}`,
        background: `linear-gradient(160deg, ${alpha(theme.palette.background.paper, 0.95)} 0%, ${alpha(
          theme.palette.background.default,
          0.9
        )} 55%, ${alpha(theme.palette.primary.light, 0.12)} 100%)`,
        boxShadow: "0 20px 45px -18px rgba(15,118,110,0.45)",
        backdropFilter: "blur(22px)"
      }}
    >
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          background: `radial-gradient(circle at 18% 12%, ${alpha(theme.palette.secondary.light, 0.32)}, transparent 55%)`,
          pointerEvents: "none"
        }}
      />
      <CardHeader
        avatar={
          <Avatar sx={{ bgcolor: alpha(theme.palette.primary.main, 0.65) }}>
            {model.identifier.charAt(0).toUpperCase()}
          </Avatar>
        }
        title={<Typography variant="h6">{model.identifier}</Typography>}
        subheader={model.description}
        sx={{ position: "relative", zIndex: 1 }}
        action={
          <Chip
            color={stateDescriptor.color}
            icon={<CheckCircleRoundedIcon fontSize="small" />}
            label={stateDescriptor.label}
            variant={state === "idle" ? "outlined" : "filled"}
          />
        }
      />
      <CardContent sx={{ position: "relative", zIndex: 1, flexGrow: 1 }}>
        <Stack spacing={2}>
          <Stack direction="row" spacing={1} flexWrap="wrap">
            <Chip
              icon={downloaded ? <CloudDoneRoundedIcon /> : <CloudDownloadRoundedIcon />}
              label={downloaded ? "Artefacts en cache" : "Téléchargement requis"}
              color={downloaded ? "success" : "default"}
              variant={downloaded ? "filled" : "outlined"}
              size="small"
            />
            <Chip
              icon={<MemoryRoundedIcon />}
              label={`Cible GPU: ${selectedDevices.length ? selectedDevices.join(", ") : "auto"}`}
              size="small"
              variant="outlined"
            />
            <Chip
              icon={<StorageRoundedIcon />}
              label={`Format: ${model.format}`}
              size="small"
              variant="outlined"
            />
          </Stack>

          <Box>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.5 }}>
              <Typography variant="subtitle2">Progression</Typography>
              <Typography variant="caption" color="text.secondary">
                {progress}%
              </Typography>
            </Stack>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 10,
                borderRadius: 6,
                backgroundColor: alpha(theme.palette.common.white, 0.15),
                "& .MuiLinearProgress-bar": {
                  borderRadius: 6,
                  background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`
                }
              }}
            />
            <Typography variant="body2" sx={{ mt: 0.5 }} color="text.secondary">
              {statusMessage}
            </Typography>
          </Box>

          <Collapse in={detailEntries.length > 0} timeout={300} unmountOnExit>
            <Stack spacing={1.5}>
              <Typography variant="subtitle2">Paramètres actifs</Typography>
              <Stack direction="row" flexWrap="wrap" spacing={1}>
                {detailEntries.map(([detailKey, detailValue]) => (
                  <Chip
                    key={detailKey}
                    size="small"
                    icon={<RouterRoundedIcon fontSize="small" />}
                    label={`${detailKey}: ${formatValue(detailValue)}`}
                    sx={{ bgcolor: alpha(theme.palette.primary.light, 0.12) }}
                  />
                ))}
              </Stack>
            </Stack>
          </Collapse>

          <Collapse in={serverEntries.length > 0} timeout={300} unmountOnExit>
            <Box>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                <LanRoundedIcon fontSize="small" />
                <Typography variant="subtitle2">Points d'accès exposés</Typography>
              </Stack>
              <List dense disablePadding>
                {serverEntries.map(([serverKey, serverValue]) => (
                  <ListItem
                    key={serverKey}
                    secondaryAction={
                      typeof serverValue === "string" || typeof serverValue === "number" ? (
                        <Tooltip title={copiedField === serverKey ? "Copié" : "Copier"}>
                          <IconButton edge="end" onClick={() => handleCopy(serverKey, serverValue)}>
                            <PlayArrowRoundedIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      ) : undefined
                    }
                    sx={{ pr: 6 }}
                  >
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: alpha(theme.palette.secondary.main, 0.15), color: theme.palette.secondary.main }}>
                        <RouterRoundedIcon fontSize="small" />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText primary={serverKey} secondary={formatValue(serverValue)} />
                  </ListItem>
                ))}
              </List>
            </Box>
          </Collapse>

          {runtime?.state === "error" && runtime.last_error && (
            <Alert severity="error" variant="outlined">
              {runtime.last_error}
            </Alert>
          )}
        </Stack>
      </CardContent>
      <Divider sx={{ opacity: 0.2 }} />
      <Box sx={{ p: 2.5, pt: 2, display: "flex", flexWrap: "wrap", gap: 1.5, justifyContent: "space-between" }}>
        <Stack direction="row" spacing={1.5}>
          <Button
            size="small"
            variant="outlined"
            startIcon={<CloudDownloadRoundedIcon />}
            onClick={onDownload}
            disabled={downloaded || state === "loading" || isLoading}
          >
            Télécharger
          </Button>
          <Button
            size="small"
            variant="contained"
            startIcon={<PlayArrowRoundedIcon />}
            onClick={onLoad}
            disabled={model.loaded || state === "loading" || isDownloading}
          >
            Charger
          </Button>
          <Button
            size="small"
            color="warning"
            variant="outlined"
            startIcon={<PowerSettingsNewRoundedIcon />}
            onClick={onUnload}
            disabled={!model.loaded || state === "loading" || isDownloading}
          >
            Décharger
          </Button>
        </Stack>
        <Select
          multiple
          displayEmpty
          size="small"
          value={selectedDevices.map((item) => String(item))}
          onChange={handleDeviceChange}
          disabled={gpuOptions.length === 0}
          renderValue={(selected) =>
            selected.length === 0
              ? "GPU auto"
              : (selected as string[])
                  .map((item) => Number(item))
                  .join(", ")
          }
          sx={{
            minWidth: 160,
            backgroundColor: alpha(theme.palette.background.paper, 0.7),
            borderRadius: 2
          }}
          MenuProps={{ disableCloseOnSelect: true }}
        >
          {gpuOptions.map((gpu) => (
            <MenuItem key={gpu.id} value={String(gpu.id)}>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ width: "100%" }}>
                <Avatar sx={{ width: 28, height: 28 }}>
                  <MemoryRoundedIcon fontSize="small" />
                </Avatar>
                <ListItemText
                  primary={`GPU ${gpu.id}`}
                  secondary={`${gpu.name} · ${gpu.memory_used.toFixed(1)}/${gpu.memory_total.toFixed(1)} Go`}
                />
              </Stack>
            </MenuItem>
          ))}
        </Select>
      </Box>
      {(isDownloading || isLoading || isUnloading) && (
        <LinearProgress
          color="secondary"
          sx={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 4 }}
        />
      )}
    </Card>
  );
};

export default ModelCard;
