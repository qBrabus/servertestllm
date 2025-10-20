import { useQuery } from "@tanstack/react-query";

import { DashboardState, fetchDashboard } from "../services/api";

export const useDashboard = () => {
  return useQuery<DashboardState>({
    queryKey: ["dashboard"],
    queryFn: fetchDashboard,
    refetchInterval: 5000
  });
};
