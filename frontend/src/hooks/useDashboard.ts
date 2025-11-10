import { useQuery } from "@tanstack/react-query";

import { DashboardState, fetchDashboard } from "../services/api";

export const useDashboard = () => {
  return useQuery<DashboardState>({
    queryKey: ["dashboard"],
    queryFn: fetchDashboard,
    refetchInterval: (query) => {
      const current = query.state.data;
      if (!current) {
        return 5000;
      }

      const models = current.models ?? {};
      const hasActiveWork = Object.values(models).some((entry) => {
        return (entry.runtime?.state ?? "idle") === "loading";
      });

      return hasActiveWork ? 1000 : 5000;
    },
    refetchIntervalInBackground: true
  });
};
