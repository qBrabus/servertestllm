import { useMemo, useState } from "react";
import {
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Stack,
  TextField,
  Typography
} from "@mui/material";

import { ChatCompletionPayload, ChatMessage, createChatCompletion } from "../services/api";

const PlaygroundPage = () => {
  const [model, setModel] = useState<string>("qwen");
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxTokens, setMaxTokens] = useState<number>(512);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "system", content: "You are a helpful assistant." }
  ]);
  const [userInput, setUserInput] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const conversation = useMemo(
    () => messages.filter((message) => message.role !== "system"),
    [messages]
  );

  const sendMessage = async () => {
    if (!userInput) return;
    const payload: ChatCompletionPayload = {
      model,
      temperature,
      max_tokens: maxTokens,
      messages: [...messages, { role: "user", content: userInput }]
    };
    setIsLoading(true);
    try {
      const data = await createChatCompletion(payload);
      const content = data.choices[0].message.content as string;
      setMessages((prev) => [
        ...prev,
        { role: "user", content: userInput },
        { role: "assistant", content }
      ]);
      setUserInput("");
    } finally {
      setIsLoading(false);
    }
  };

  const resetConversation = () => {
    setMessages([{ role: "system", content: "You are a helpful assistant." }]);
    setUserInput("");
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        OpenAI-Compatible Playground
      </Typography>
      <Stack direction={{ xs: "column", md: "row" }} spacing={3}>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Request Builder
            </Typography>
            <Stack spacing={2}>
              <TextField select label="Model" value={model} onChange={(e) => setModel(e.target.value)}>
                <MenuItem value="qwen">Qwen 30B Instruct</MenuItem>
              </TextField>
              <TextField
                label="Temperature"
                type="number"
                value={temperature}
                inputProps={{ min: 0, max: 2, step: 0.1 }}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
              <TextField
                label="Max Tokens"
                type="number"
                value={maxTokens}
                inputProps={{ min: 1, max: 4096 }}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />
              <TextField
                label="User Prompt"
                multiline
                minRows={6}
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
              />
              <Stack direction="row" spacing={2}>
                <Button variant="contained" onClick={sendMessage} disabled={isLoading}>
                  {isLoading ? "Generating..." : "Send"}
                </Button>
                <Button variant="outlined" color="inherit" onClick={resetConversation}>
                  Reset
                </Button>
              </Stack>
            </Stack>
          </CardContent>
        </Card>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Conversation
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <List sx={{ maxHeight: 520, overflow: "auto" }}>
              {conversation.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  Interactions will appear here once you send a message.
                </Typography>
              )}
              {conversation.map((message, index) => (
                <ListItem key={`${message.role}-${index}`} alignItems="flex-start">
                  <ListItemText
                    primary={message.role === "user" ? "You" : "Assistant"}
                    secondary={message.content}
                    secondaryTypographyProps={{
                      component: "span",
                      sx: { whiteSpace: "pre-wrap" }
                    }}
                  />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      </Stack>
    </Box>
  );
};

export default PlaygroundPage;
