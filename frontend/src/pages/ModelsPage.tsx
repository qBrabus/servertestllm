import {
  Alert,
  Box,
  Button,
  Card,
  CardActions,
  CardContent,
  Checkbox,
  Chip,
  FormControl,
  Grid,
  InputLabel,
  LinearProgress,
  ListItemText,
  MenuItem,
  OutlinedInput,
  Select,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  fetchHuggingFaceTokenStatus,
  HuggingFaceTokenStatus,
  loadModel,
  unloadModel,
  updateHuggingFaceToken
} from "../services/api";
import { useDashboard } from "../hooks/useDashboard";

const ModelsPage = () => {
  const queryClient = useQueryClient();
  const { data, isLoading } = useDashboard();
  const [selectedDevices, setSelectedDevices] = useState<Record<string, number[]>>({});
  const [tokenInput, setTokenInput] = useState("");
  const [tokenFeedback, setTokenFeedback] = useState<
    { message: string; severity: "success" | "error" } | null
  >(null);

  const tokenQuery = useQuery<HuggingFaceTokenStatus>({
    queryKey: ["huggingface-token"],
    queryFn: fetchHuggingFaceTokenStatus,
    refetchInterval: 60000
  });

  const loadMutation = useMutation({
    mutationFn: loadModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["dashboard"] })
  });

  const unloadMutation = useMutation({
    mutationFn: unloadModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["dashboard"] })
  });

  const tokenMutation = useMutation<HuggingFaceTokenStatus, Error, string | null>({
    mutationFn: updateHuggingFaceToken,
    onSuccess: (result, _value) => {
      queryClient.invalidateQueries({ queryKey: ["huggingface-token"] });
      setTokenInput("");
      setTokenFeedback({
        message: result.has_token
          ? "Hugging Face token saved successfully."
          : "Hugging Face token removed.",
        severity: "success"
      });
    },
    onError: (error) => {
      setTokenFeedback({
        message: error?.message ?? "Unable to update Hugging Face token.",
        severity: "error"
      });
    }
  });

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Model Orchestration
      </Typography>
      {(isLoading ||
        loadMutation.isPending ||
        unloadMutation.isPending ||
        tokenMutation.isPending ||
        tokenQuery.isLoading) && <LinearProgress />}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6">Hugging Face Access Token</Typography>
              <Typography variant="body2" color="text.secondary">
                Provide your Hugging Face personal access token so that gated models can be
                downloaded directly from the server.
              </Typography>
              <TextField
                type="password"
                label="Token"
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
                    ? "A token is already stored on the server"
                    : "Enter your Hugging Face token"
                }
                helperText={
                  tokenQuery.data?.has_token
                    ? "A token is stored. Submitting a new one will overwrite it."
                    : "No token stored yet."
                }
              />
              <Stack
                direction={{ xs: "column", sm: "row" }}
                spacing={2}
                sx={{ mt: 2 }}
              >
                <Button
                  variant="contained"
                  onClick={() => tokenMutation.mutate(tokenInput.trim())}
                  disabled={tokenInput.trim().length === 0 || tokenMutation.isPending}
                >
                  Save token
                </Button>
                <Button
                  variant="outlined"
                  color="warning"
                  onClick={() => tokenMutation.mutate(null)}
                  disabled={
                    !tokenQuery.data?.has_token || tokenMutation.isPending || tokenQuery.isLoading
                  }
                >
                  Clear stored token
                </Button>
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
        {data &&
          Object.entries(data.models).map(([key, model]) => {
            const activeDeviceIds = Array.isArray(model.params?.device_ids)
              ? (model.params?.device_ids as number[]).map((value) => Number(value))
              : [];
            const selectedForModel = selectedDevices[key] ?? activeDeviceIds;
            const gpuOptions = data.gpus ?? [];

            const handleDeviceChange = (event: SelectChangeEvent<string[]>) => {
              const value = event.target.value;
              const normalized = (typeof value === "string" ? value.split(",") : value)
                .filter((item) => item !== "")
                .map((item) => Number(item));
              setSelectedDevices((prev) => ({ ...prev, [key]: normalized }));
            };

            return (
              <Grid item xs={12} md={4} key={key}>
                <Card>
                  <CardContent>
                    <Typography variant="h6">{model.identifier}</Typography>
                    <Typography variant="body2" color="text.secondary">
                    {model.description}
                  </Typography>
                  <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap" }}>
                    <Chip label={model.task} color="primary" />
                    <Chip
                      label={model.loaded ? "Loaded" : "Not Loaded"}
                      color={model.loaded ? "success" : "default"}
                    />
                  </Stack>
                  <Typography variant="caption" sx={{ mt: 1, display: "block" }}>
                    Format: {model.format}
                  </Typography>
                  <Typography variant="caption" sx={{ display: "block" }}>
                    Active GPUs:{" "}
                    {activeDeviceIds.length > 0 ? activeDeviceIds.join(", ") : "Auto"}
                  </Typography>
                  <FormControl fullWidth size="small" sx={{ mt: 2 }} disabled={gpuOptions.length === 0}>
                    <InputLabel id={`gpu-select-${key}`}>Select GPUs</InputLabel>
                    <Select
                      multiple
                      displayEmpty
                      labelId={`gpu-select-${key}`}
                      value={selectedForModel.map((item) => String(item))}
                      onChange={handleDeviceChange}
                      input={<OutlinedInput label="Select GPUs" />}
                      renderValue={(selected) =>
                        selected.length === 0
                          ? "Auto"
                          : (selected as string[])
                              .map((item) => Number(item))
                              .join(", ")
                      }
                    >
                      {gpuOptions.map((gpu) => (
                        <MenuItem key={gpu.id} value={String(gpu.id)}>
                          <Checkbox checked={selectedForModel.includes(gpu.id)} />
                          <ListItemText primary={`GPU ${gpu.id} (${gpu.name})`} />
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </CardContent>
                <CardActions>
                  <Button
                    size="small"
                    variant="contained"
                    onClick={() =>
                      loadMutation.mutate({ key, gpuDeviceIds: selectedDevices[key] })
                    }
                    disabled={model.loaded}
                  >
                    Load
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    color="warning"
                    onClick={() => unloadMutation.mutate(key)}
                    disabled={!model.loaded}
                  >
                    Unload
                  </Button>
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
