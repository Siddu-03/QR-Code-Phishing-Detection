import { useState } from "react";
import {
  FaBell,
  FaCamera,
  FaShieldAlt,
  FaExclamationTriangle,
  FaVolumeUp,
  FaTrash,
  FaFileExport,
  FaInfoCircle,
} from "react-icons/fa";

export default function Settings() {
  const [notifications, setNotifications] = useState(true);
  const [autoScan, setAutoScan] = useState(true);
  const [safeBrowsing, setSafeBrowsing] = useState(true);
  const [warnLinks, setWarnLinks] = useState(true);
  const [scanSound, setScanSound] = useState(true);

  return (
    <div className="min-h-screen bg-gray-100 py-10 px-4">
      <div className="max-w-3xl mx-auto">

        <h1 className="text-4xl font-bold text-center mb-8">
          ⚙️ Settings
        </h1>

        {/* Security Settings */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">

          <h2 className="text-2xl font-semibold mb-5">
            Security & Scanner
          </h2>

          <SettingToggle
            icon={<FaBell />}
            title="Enable Notifications"
            checked={notifications}
            onChange={() => setNotifications(!notifications)}
          />

          <SettingToggle
            icon={<FaCamera />}
            title="Auto Scan"
            checked={autoScan}
            onChange={() => setAutoScan(!autoScan)}
          />

          <SettingToggle
            icon={<FaShieldAlt />}
            title="Safe Browsing Alerts"
            checked={safeBrowsing}
            onChange={() => setSafeBrowsing(!safeBrowsing)}
          />

          <SettingToggle
            icon={<FaExclamationTriangle />}
            title="Warn Before Opening Suspicious Links"
            checked={warnLinks}
            onChange={() => setWarnLinks(!warnLinks)}
          />

          <SettingToggle
            icon={<FaVolumeUp />}
            title="Scan Sound"
            checked={scanSound}
            onChange={() => setScanSound(!scanSound)}
          />

        </div>

        {/* Data */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">

          <h2 className="text-2xl font-semibold mb-5">
            Data
          </h2>

          <button
            className="w-full bg-red-500 hover:bg-red-600 text-white py-3 rounded-xl mb-4 flex items-center justify-center gap-2 transition"
          >
            <FaTrash />
            Clear Scan History
          </button>

          <button
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-xl flex items-center justify-center gap-2 transition"
          >
            <FaFileExport />
            Export History
          </button>

        </div>

        {/* About */}
        <div className="bg-white rounded-2xl shadow-lg p-6">

          <h2 className="text-2xl font-semibold mb-5">
            About
          </h2>

          <div className="flex justify-between items-center">

            <div className="flex items-center gap-3">
              <FaInfoCircle className="text-blue-600" />
              <span className="font-medium">
                Version
              </span>
            </div>

            <span className="text-gray-600">
              v1.0.0
            </span>

          </div>

        </div>

      </div>
    </div>
  );
}

type SettingToggleProps = {
  icon: React.ReactNode;
  title: string;
  checked: boolean;
  onChange: () => void;
};

function SettingToggle({
  icon,
  title,
  checked,
  onChange,
}: SettingToggleProps) {
  return (
    <div className="flex justify-between items-center py-4 border-b last:border-none">

      <div className="flex items-center gap-3 text-lg">
        {icon}
        <span>{title}</span>
      </div>

      <button
        onClick={onChange}
        className={`w-14 h-8 rounded-full transition relative ${
          checked ? "bg-blue-600" : "bg-gray-300"
        }`}
      >
        <div
          className={`absolute top-1 w-6 h-6 bg-white rounded-full transition ${
            checked ? "left-7" : "left-1"
          }`}
        />
      </button>

    </div>
  );
}