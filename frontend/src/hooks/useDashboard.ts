import { useQuery } from "@tanstack/react-query";

import { DashboardState, fetchDashboard } from "../services/api";

export const useDashboard = () => {
  return useQuery<DashboardState>({
    queryKey: ["dashboard"],
    queryFn: fetchDashboard,
    refetchInterval: (data) => {
      if (!data) {
        return 5000;
      }

      const models = data.models ?? {};
      const hasActiveWork = Object.values(models).some((model) => {
        return model.runtime?.state === "loading";
      });

      return hasActiveWork ? 1000 : 5000;
    },
    refetchIntervalInBackground: true
  });
};
