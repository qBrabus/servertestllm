import { useState } from "react";
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  TextField,
  Tooltip
} from "@mui/material";
import VpnKeyIcon from "@mui/icons-material/VpnKey";

import { getStoredApiKey, setStoredApiKey } from "../services/api";

const ApiKeyDialog = () => {
  const [open, setOpen] = useState(false);
  const [apiKey, setApiKey] = useState<string>(getStoredApiKey() ?? "");

  const handleSave = () => {
    setStoredApiKey(apiKey || null);
    setOpen(false);
  };

  const handleClear = () => {
    setApiKey("");
    setStoredApiKey(null);
    setOpen(false);
  };

  return (
    <>
      <Tooltip title="Configure API key for secured endpoints">
        <IconButton color="inherit" onClick={() => setOpen(true)}>
          <VpnKeyIcon />
        </IconButton>
      </Tooltip>
      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>OpenAI-Compatible API Key</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Provide the API key configured on the server (environment variable <code>OPENAI_KEYS</code>).
            The key is stored locally in your browser and attached to every API request as a Bearer token.
          </DialogContentText>
          <TextField
            label="API Key"
            fullWidth
            margin="dense"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClear}>Clear</Button>
          <Button variant="contained" onClick={handleSave}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ApiKeyDialog;
