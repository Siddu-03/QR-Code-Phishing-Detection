import { Scanner } from "@yudiel/react-qr-scanner";

interface Props {
  onScan: (text: string) => void;
}

export default function CameraScanner({ onScan }: Props) {
  return (
    <div className="rounded-xl overflow-hidden shadow-lg">
      <Scanner
        onScan={(results) => {
          if (results.length > 0) {
            onScan(results[0].rawValue);
          }
        }}
        onError={(error) => {
          console.error(error);
        }}
      />
    </div>
  );
}