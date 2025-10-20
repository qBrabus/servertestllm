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
  Typography
} from "@mui/material";
import { useState } from "react";
import { Route, Routes, useLocation, useNavigate } from "react-router-dom";

import DashboardPage from "./pages/DashboardPage";
import ModelsPage from "./pages/ModelsPage";
import PlaygroundPage from "./pages/PlaygroundPage";
import AudioPage from "./pages/AudioPage";
import ApiKeyDialog from "./components/ApiKeyDialog";

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
      <Toolbar>
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
    <Box sx={{ display: "flex" }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
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
          <SettingsEthernetIcon sx={{ mr: 1 }} />
          <Typography variant="h6" noWrap component="div">
            Unified Inference Control Plane
          </Typography>
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
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: 8
        }}
      >
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/audio" element={<AudioPage />} />
          <Route path="/playground" element={<PlaygroundPage />} />
        </Routes>
      </Box>
    </Box>
  );
};

export default App;
