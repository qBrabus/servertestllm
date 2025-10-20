import { ChangeEvent, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  Grid,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Typography
} from "@mui/material";

import { diarizeAudio, transcribeAudio, DiarizationResultSegment, TranscriptionResult } from "../services/api";

const AudioPage = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [transcription, setTranscription] = useState<TranscriptionResult | null>(null);
  const [diarization, setDiarization] = useState<DiarizationResultSegment[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>, task: "asr" | "diarization") => {
    const files = event.target.files;
    if (!files || files.length === 0) {
      return;
    }
    setIsLoading(true);
    setError(null);
    const file = files[0];
    try {
      if (task === "asr") {
        const result = await transcribeAudio(file);
        setTranscription(result);
      } else {
        const result = await diarizeAudio(file);
        setDiarization(result.segments);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
      event.target.value = "";
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Audio Intelligence
      </Typography>
      {isLoading && <LinearProgress sx={{ mb: 2 }} />}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">Automatic Speech Recognition</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Upload a WAV/FLAC/OGG file to obtain a transcript using NVIDIA Canary.
              </Typography>
              <Button variant="contained" component="label">
                Upload Audio
                <input hidden type="file" accept="audio/*" onChange={(event) => handleUpload(event, "asr")} />
              </Button>
              <Divider sx={{ my: 2 }} />
              {transcription ? (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Transcript (sample rate {transcription.sampling_rate} Hz)
                  </Typography>
                  <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
                    {transcription.text}
                  </Typography>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No transcription yet.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">Speaker Diarization</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Upload an audio clip to separate speakers with Pyannote.
              </Typography>
              <Button variant="contained" component="label" color="secondary">
                Upload Audio
                <input hidden type="file" accept="audio/*" onChange={(event) => handleUpload(event, "diarization")} />
              </Button>
              <Divider sx={{ my: 2 }} />
              {diarization && diarization.length > 0 ? (
                <List>
                  {diarization.map((segment, index) => (
                    <ListItem key={index} divider>
                      <ListItemText
                        primary={`Speaker ${segment.speaker}`}
                        secondary={`Start: ${segment.start.toFixed(2)}s · End: ${segment.end.toFixed(2)}s`}
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Upload audio to view diarization segments.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AudioPage;
