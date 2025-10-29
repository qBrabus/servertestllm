import { Box, Card, CardContent, Grid, LinearProgress, Typography } from "@mui/material";
import { useMemo } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

import { useDashboard } from "../hooks/useDashboard";
import TorchStackCard from "../components/TorchStackCard";

const DashboardPage = () => {
  const { data, isLoading } = useDashboard();

  const gpuData = useMemo(() => {
    if (!data?.gpus) return [];
    return data.gpus.map((gpu) => ({
      name: `GPU ${gpu.id}`,
      load: Math.round(gpu.load * 100),
      memory: Number(((gpu.memory_used / gpu.memory_total) * 100).toFixed(2))
    }));
  }, [data]);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Infrastructure Overview
      </Typography>
      {isLoading && <LinearProgress />}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">CPU Usage</Typography>
              <Typography variant="h3">
                {data ? `${data.system.cpu_percent.toFixed(1)}%` : "--"}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={data ? data.system.cpu_percent : 0}
                sx={{ mt: 2 }}
              />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">Memory Usage</Typography>
              <Typography variant="h3">
                {data ? `${data.system.memory_percent.toFixed(1)}%` : "--"}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={data ? data.system.memory_percent : 0}
                sx={{ mt: 2 }}
              />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <TorchStackCard dependencies={data?.dependencies ?? []} />
        </Grid>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                GPU Utilization
              </Typography>
              <Box sx={{ height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={gpuData}>
                    <XAxis dataKey="name" stroke="#fff" />
                    <YAxis stroke="#fff" />
                    <Tooltip />
                    <Bar dataKey="load" fill="#60a5fa" name="Compute Load %" />
                    <Bar dataKey="memory" fill="#fbbf24" name="Memory Usage %" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
