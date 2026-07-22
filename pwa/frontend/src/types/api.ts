export interface ScanRequest {
  qr_data: string;
}

export interface ScanResponse {
  risk: "Safe" | "Suspicious" | "Danger";
  message: string;
  url?: string;
}

export interface HistoryItem {
  id: number;
  url: string;
  risk: string;
  timestamp: string;
}