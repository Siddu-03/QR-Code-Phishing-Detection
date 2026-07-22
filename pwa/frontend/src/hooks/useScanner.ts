import { useState } from "react";
import { scanQRCode } from "../services/scanner";
import type { ScanResponse } from "../types/api";

export function useScanner() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScanResponse | null>(null);
  const [error, setError] = useState("");

  const scan = async (qrData: string) => {
    try {
      setLoading(true);
      setError("");

      const response = await scanQRCode({
        qr_data: qrData,
      });

      setResult(response);
    } catch (err) {
      console.error(err);
      setError("Unable to scan QR code.");
    } finally {
      setLoading(false);
    }
  };

  return {
    loading,
    result,
    error,
    scan,
  };
}