import ArticleRoundedIcon from "@mui/icons-material/ArticleRounded";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import CloudDoneRoundedIcon from "@mui/icons-material/CloudDoneRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import ContentCopyRoundedIcon from "@mui/icons-material/ContentCopyRounded";
import DnsRoundedIcon from "@mui/icons-material/DnsRounded";
import HttpRoundedIcon from "@mui/icons-material/HttpRounded";
import LanRoundedIcon from "@mui/icons-material/LanRounded";
import LinkRoundedIcon from "@mui/icons-material/LinkRounded";
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
  IconButton,
  LinearProgress,
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
import { ReactNode, useMemo, useState } from "react";

import { GPUInfo, ModelInfo, ModelRuntimeInfo } from "../services/api";

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
  if (typeof value === "boolean") {
    return value ? "Oui" : "Non";
  }
  return String(value);
};

const stateLabels: Record<
  ModelRuntimeInfo["state"],
  { label: string; color: "default" | "info" | "success" | "error" }
> = {
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
  const downloaded = model.loaded || runtime?.downloaded || false;
  const progress = Math.min(
    100,
    runtime?.progress ?? (downloaded ? 100 : isDownloading ? 8 : 0)
  );
  const statusMessage =
    runtime?.status ??
    (state === "loading"
      ? "Initialisation en cours..."
      : downloaded
        ? model.loaded
          ? "Modèle prêt"
          : "Artefacts disponibles"
        : "En attente de chargement");

  const serverInfo = runtime?.server as Record<string, unknown> | undefined;
  const detailsInfo = runtime?.details as Record<string, unknown> | undefined;

  const serverHighlights = useMemo(
    () => {
      if (!serverInfo) {
        return [] as Array<{ key: string; label: string; value: string; icon: ReactNode }>;
      }

      const highlights: Array<{ key: string; label: string; value: string; icon: ReactNode }> = [];
      const url = serverInfo.url;
      if (typeof url === "string" && url) {
        highlights.push({
          key: "url",
          label: "URL",
          value: url,
          icon: <LinkRoundedIcon fontSize="small" />
        });
      }

      const endpoint = serverInfo.endpoint;
      if (typeof endpoint === "string" && endpoint) {
        highlights.push({
          key: "endpoint",
          label: "Endpoint",
          value: endpoint,
          icon: <HttpRoundedIcon fontSize="small" />
        });
      }

      const hostValue = typeof serverInfo.host === "string" ? serverInfo.host : null;
      const portValue =
        typeof serverInfo.port === "number" || typeof serverInfo.port === "string"
          ? String(serverInfo.port)
          : null;
      if (hostValue) {
        highlights.push({
          key: "host",
          label: "Hôte",
          value: portValue ? `${hostValue}:${portValue}` : hostValue,
          icon: <DnsRoundedIcon fontSize="small" />
        });
      }

      const docs = serverInfo.docs;
      if (typeof docs === "string" && docs && docs !== url) {
        highlights.push({
          key: "docs",
          label: "Documentation",
          value: docs,
          icon: <ArticleRoundedIcon fontSize="small" />
        });
      }

      const type = serverInfo.type;
      if (typeof type === "string" && type) {
        highlights.push({
          key: "type",
          label: "Type",
          value: type,
          icon: <LanRoundedIcon fontSize="small" />
        });
      }

      const device = serverInfo.device;
      if (typeof device === "string" && device) {
        highlights.push({
          key: "device",
          label: "Périphérique",
          value: device,
          icon: <MemoryRoundedIcon fontSize="small" />
        });
      }

      return highlights;
    },
    [serverInfo]
  );

  const serverExtraEntries = useMemo(
    () => {
      if (!serverInfo) {
        return [] as Array<[string, unknown]>;
      }
      const highlightKeys = new Set(serverHighlights.map((item) => item.key));
      return Object.entries(serverInfo).filter(([key, value]) => {
        if (highlightKeys.has(key)) {
          return false;
        }
        if (value === null || value === undefined) {
          return false;
        }
        if (typeof value === "object") {
          return false;
        }
        return true;
      });
    },
    [serverInfo, serverHighlights]
  );

  const detailEntries = useMemo(
    () => {
      if (!detailsInfo) {
        return [] as Array<[string, unknown]>;
      }
      return Object.entries(detailsInfo).filter(([, value]) => value !== undefined && value !== null);
    },
    [detailsInfo]
  );

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

  const downloadChipLabel = downloaded
    ? model.loaded
      ? "Modèle chargé"
      : "Artefacts en cache"
    : state === "loading" || isDownloading
      ? "Téléchargement en cours"
      : "Téléchargement requis";

  const downloadChipColor: "default" | "info" | "success" = downloaded
    ? "success"
    : state === "loading" || isDownloading
      ? "info"
      : "default";

  return (
    <Card
      key={modelKey}
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        borderRadius: 3,
        border: `1px solid ${alpha(theme.palette.divider, 0.6)}`,
        backgroundColor: alpha(theme.palette.background.paper, 0.92),
        backdropFilter: "blur(14px)",
        boxShadow: "0 18px 34px -22px rgba(15,118,110,0.55)"
      }}
    >
      <CardHeader
        avatar={
          <Avatar sx={{ bgcolor: alpha(theme.palette.primary.main, 0.65) }}>
            {model.identifier.charAt(0).toUpperCase()}
          </Avatar>
        }
        title={
          <Typography variant="subtitle1" fontWeight={600} noWrap>
            {model.identifier}
          </Typography>
        }
        subheader={
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}
          >
            {model.description}
          </Typography>
        }
        sx={{
          px: 2.5,
          pt: 2.25,
          pb: 0.75,
          "& .MuiCardHeader-action": { alignSelf: "center" }
        }}
        action={
          <Chip
            color={stateDescriptor.color}
            icon={<CheckCircleRoundedIcon fontSize="small" />}
            label={stateDescriptor.label}
            size="small"
            variant={state === "idle" ? "outlined" : "filled"}
          />
        }
      />
      <CardContent sx={{ flexGrow: 1, px: 2.5, pt: 1.5, pb: 2.25 }}>
        <Stack spacing={2} sx={{ height: "100%" }}>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            <Chip
              icon={downloaded ? <CloudDoneRoundedIcon fontSize="small" /> : <CloudDownloadRoundedIcon fontSize="small" />}
              label={downloadChipLabel}
              color={downloadChipColor}
              variant={downloaded ? "filled" : "outlined"}
              size="small"
            />
            <Chip
              icon={<MemoryRoundedIcon fontSize="small" />}
              label={`Cible GPU : ${selectedDevices.length ? selectedDevices.join(", ") : "auto"}`}
              size="small"
              variant="outlined"
            />
            <Chip
              icon={<StorageRoundedIcon fontSize="small" />}
              label={`Format : ${model.format}`}
              size="small"
              variant="outlined"
            />
          </Stack>

          <Box>
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.75 }}>
              <Typography variant="subtitle2">Progression</Typography>
              <Typography variant="caption" color="text.secondary">
                {progress}%
              </Typography>
            </Stack>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 8,
                borderRadius: 6,
                backgroundColor: alpha(theme.palette.text.disabled, 0.1),
                "& .MuiLinearProgress-bar": {
                  borderRadius: 6,
                  background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`
                }
              }}
            />
            <Typography variant="caption" sx={{ mt: 0.75 }} color="text.secondary">
              {statusMessage}
            </Typography>
          </Box>

          <Collapse in={detailEntries.length > 0} timeout={200} unmountOnExit>
            <Stack spacing={1}>
              <Typography variant="subtitle2">Paramètres actifs</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
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

          <Collapse
            in={serverHighlights.length > 0 || serverExtraEntries.length > 0}
            timeout={200}
            unmountOnExit
          >
            <Stack spacing={1.25}>
              <Typography variant="subtitle2">Points d'accès exposés</Typography>
              {serverHighlights.map(({ key, label, value, icon }) => (
                <Stack
                  key={key}
                  direction="row"
                  spacing={1}
                  alignItems="center"
                  sx={{
                    p: 1,
                    borderRadius: 2,
                    bgcolor: alpha(theme.palette.primary.main, 0.06)
                  }}
                >
                  <Avatar
                    sx={{
                      bgcolor: alpha(theme.palette.primary.main, 0.18),
                      color: theme.palette.primary.main,
                      width: 32,
                      height: 32
                    }}
                  >
                    {icon}
                  </Avatar>
                  <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                    <Typography variant="caption" color="text.secondary">
                      {label}
                    </Typography>
                    <Typography variant="body2" noWrap title={value}>
                      {value}
                    </Typography>
                  </Box>
                  <Tooltip title={copiedField === key ? "Copié" : "Copier"}>
                    <IconButton size="small" onClick={() => handleCopy(key, value)}>
                      <ContentCopyRoundedIcon fontSize="inherit" />
                    </IconButton>
                  </Tooltip>
                </Stack>
              ))}
              {serverExtraEntries.length > 0 && (
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  {serverExtraEntries.map(([extraKey, extraValue]) => (
                    <Chip
                      key={extraKey}
                      size="small"
                      icon={<RouterRoundedIcon fontSize="small" />}
                      label={`${extraKey}: ${formatValue(extraValue)}`}
                      sx={{ bgcolor: alpha(theme.palette.secondary.main, 0.1) }}
                    />
                  ))}
                </Stack>
              )}
            </Stack>
          </Collapse>

          {runtime?.state === "error" && runtime.last_error && (
            <Alert severity="error" variant="outlined">
              {runtime.last_error}
            </Alert>
          )}
        </Stack>
      </CardContent>
      <Box
        sx={{
          px: 2.5,
          py: 1.75,
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 1.2,
          justifyContent: "space-between"
        }}
      >
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
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
            backgroundColor: alpha(theme.palette.background.paper, 0.75),
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
          sx={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 3 }}
        />
      )}
    </Card>
  );
};

export default ModelCard;
