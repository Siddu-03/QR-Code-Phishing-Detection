import api from "./api";
import type { ScanRequest, ScanResponse } from "../types/api";

export async function scanQRCode(
  data: ScanRequest
): Promise<ScanResponse> {
  const response = await api.post("/scan", data);
  return response.data;
}

export async function getHistory() {
  const response = await api.get("/history");
  return response.data;
}

export async function getHealth() {
  const response = await api.get("/health");
  return response.data;
}