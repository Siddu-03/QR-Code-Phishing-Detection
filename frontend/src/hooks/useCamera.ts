import { useState } from "react";

export function useCamera() {
  const [cameraEnabled, setCameraEnabled] = useState(false);

  const startCamera = () => setCameraEnabled(true);
  const stopCamera = () => setCameraEnabled(false);

  return {
    cameraEnabled,
    startCamera,
    stopCamera,
  };
}