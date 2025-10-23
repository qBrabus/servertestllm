import {
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
  Typography
} from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { loadModel, unloadModel } from "../services/api";
import { useDashboard } from "../hooks/useDashboard";

const ModelsPage = () => {
  const queryClient = useQueryClient();
  const { data, isLoading } = useDashboard();
  const [selectedDevices, setSelectedDevices] = useState<Record<string, number[]>>({});

  const loadMutation = useMutation({
    mutationFn: loadModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["dashboard"] })
  });

  const unloadMutation = useMutation({
    mutationFn: unloadModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["dashboard"] })
  });

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Model Orchestration
      </Typography>
      {(isLoading || loadMutation.isPending || unloadMutation.isPending) && <LinearProgress />}
      <Grid container spacing={3}>
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
