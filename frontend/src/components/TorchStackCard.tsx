import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import InfoIcon from "@mui/icons-material/Info";
import WarningIcon from "@mui/icons-material/Warning";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography
} from "@mui/material";

import type { DependencyStatus } from "../services/api";

interface TorchStackCardProps {
  dependencies: DependencyStatus[];
}

const STATUS_ICONS = {
  success: <CheckCircleIcon color="success" fontSize="small" />,
  warning: <WarningIcon color="warning" fontSize="small" />,
  error: <ErrorIcon color="error" fontSize="small" />,
  info: <InfoIcon color="info" fontSize="small" />
};

const TorchStackCard = ({ dependencies }: TorchStackCardProps) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          PyTorch Runtime Validation
        </Typography>
        {dependencies.length === 0 ? (
          <Typography color="text.secondary">
            Aucun module PyTorch n'a été détecté sur le serveur.
          </Typography>
        ) : (
          <List disablePadding>
            {dependencies.map((dependency, index) => {
              const hasError = Boolean(dependency.error);
              const isCuda = dependency.cuda === true;
              const hasDetails = dependency.details && Object.keys(dependency.details).length > 0;

              let icon = STATUS_ICONS.info;
              if (hasError) {
                icon = STATUS_ICONS.error;
              } else if (isCuda) {
                icon = STATUS_ICONS.success;
              } else if (dependency.cuda === false) {
                icon = STATUS_ICONS.warning;
              }

              return (
                <Box key={dependency.name + index.toString()}>
                  <ListItem alignItems="flex-start">
                    <ListItemIcon>{icon}</ListItemIcon>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="subtitle1">{dependency.name}</Typography>
                          {dependency.version && <Chip size="small" label={dependency.version} />}
                          {typeof dependency.cuda === "boolean" && (
                            <Chip
                              size="small"
                              color={dependency.cuda ? "success" : "default"}
                              label={dependency.cuda ? "CUDA activé" : "CUDA indisponible"}
                            />
                          )}
                        </Box>
                      }
                      secondaryTypographyProps={{ component: "div" }}
                      secondary={
                        <>
                          {dependency.error && (
                            <Typography color="error" variant="body2">
                              {dependency.error}
                            </Typography>
                          )}
                          {!dependency.error && !isCuda && dependency.cuda === false && (
                            <Typography color="warning.main" variant="body2">
                              Le module est installé mais ne détecte pas d'exécution CUDA.
                            </Typography>
                          )}
                          {hasDetails && (
                            <Box mt={1} display="flex" flexWrap="wrap" gap={1}>
                              {Object.entries(dependency.details ?? {}).map(([key, value]) => (
                                <Chip
                                  key={key}
                                  size="small"
                                  label={`${key}: ${value ?? "n/a"}`}
                                  variant="outlined"
                                />
                              ))}
                            </Box>
                          )}
                        </>
                      }
                    />
                  </ListItem>
                  {index < dependencies.length - 1 && <Divider component="li" />}
                </Box>
              );
            })}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default TorchStackCard;

