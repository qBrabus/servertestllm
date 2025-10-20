import {
  Box,
  Button,
  Card,
  CardActions,
  CardContent,
  Chip,
  Grid,
  LinearProgress,
  Stack,
  Typography
} from "@mui/material";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { loadModel, unloadModel } from "../services/api";
import { useDashboard } from "../hooks/useDashboard";

const ModelsPage = () => {
  const queryClient = useQueryClient();
  const { data, isLoading } = useDashboard();

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
          Object.entries(data.models).map(([key, model]) => (
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
                </CardContent>
                <CardActions>
                  <Button
                    size="small"
                    variant="contained"
                    onClick={() => loadMutation.mutate(key)}
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
          ))}
      </Grid>
    </Box>
  );
};

export default ModelsPage;
