import DashboardIcon from "@mui/icons-material/Dashboard";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";
import HubIcon from "@mui/icons-material/Hub";
import MenuIcon from "@mui/icons-material/Menu";
import SettingsEthernetIcon from "@mui/icons-material/SettingsEthernet";
import SurroundSoundIcon from "@mui/icons-material/SurroundSound";
import {
  AppBar,
  Box,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  alpha
} from "@mui/material";
import { useState } from "react";
import { Route, Routes, useLocation, useNavigate } from "react-router-dom";

import DashboardPage from "./pages/DashboardPage";
import ModelsPage from "./pages/ModelsPage";
import PlaygroundPage from "./pages/PlaygroundPage";
import AudioPage from "./pages/AudioPage";
import ApiKeyDialog from "./components/ApiKeyDialog";
import controlPlaneLogo from "./assets/unified-logo.svg";

const drawerWidth = 260;

const App = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { label: "Dashboard", icon: <DashboardIcon />, path: "/" },
    { label: "Models", icon: <HubIcon />, path: "/models" },
    { label: "Audio", icon: <SurroundSoundIcon />, path: "/audio" },
    { label: "Playground", icon: <GraphicEqIcon />, path: "/playground" }
  ];

  const drawer = (
    <div>
      <Toolbar sx={{ gap: 1, alignItems: "center" }}>
        <SettingsEthernetIcon color="primary" />
        <Typography variant="h6">Unified Gateway</Typography>
      </Toolbar>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => {
                navigate(item.path);
                setMobileOpen(false);
              }}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Box
      sx={{
        display: "flex",
        minHeight: "100vh",
        background:
          "radial-gradient(circle at 20% 20%, rgba(14,165,233,0.12), transparent 55%), linear-gradient(160deg, rgba(13,17,23,0.95) 0%, rgba(16,24,39,0.92) 100%)"
      }}
    >
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          background: (theme) =>
            `linear-gradient(120deg, ${alpha(theme.palette.primary.main, 0.85)} 0%, ${alpha(theme.palette.secondary.main, 0.65)} 100%)`,
          boxShadow: "0 12px 30px rgba(15,118,110,0.35)",
          backdropFilter: "blur(12px)"
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: "none" } }}
          >
            <MenuIcon />
          </IconButton>
          <Box
            component="img"
            src={controlPlaneLogo}
            alt="Unified control plane"
            sx={{ width: 40, height: 40, mr: 2, filter: "drop-shadow(0 6px 12px rgba(15,118,110,0.45))" }}
          />
          <Box>
            <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 700 }}>
              Unified Inference Control Plane
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.9 }}>
              Supervision en direct des modèles et pipelines audio/LLM
            </Typography>
          </Box>
          <Box sx={{ flexGrow: 1 }} />
          <ApiKeyDialog />
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        aria-label="navigation menu"
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: "block", sm: "none" },
            "& .MuiDrawer-paper": { boxSizing: "border-box", width: drawerWidth }
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: "none", sm: "block" },
            "& .MuiDrawer-paper": { boxSizing: "border-box", width: drawerWidth }
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 3, md: 4 },
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: 8,
          position: "relative"
        }}
      >
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            background: "radial-gradient(circle at 85% 15%, rgba(147,51,234,0.18), transparent 60%)",
            pointerEvents: "none"
          }}
        />
        <Box sx={{ position: "relative" }}>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/audio" element={<AudioPage />} />
            <Route path="/playground" element={<PlaygroundPage />} />
          </Routes>
        </Box>
      </Box>
    </Box>
  );
};

export default App;
