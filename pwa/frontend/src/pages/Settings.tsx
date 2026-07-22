import { useState } from "react";
import {
  FaMoon,
  FaBell,
  FaCamera,
  FaShieldAlt,
  FaExclamationTriangle,
  FaTrash,
  FaFileExport,
  FaInfoCircle,
  FaUserFriends,
} from "react-icons/fa";

export default function Settings() {
  const [darkMode, setDarkMode] = useState(false);
  const [notifications, setNotifications] = useState(true);
  const [autoScan, setAutoScan] = useState(true);
  const [safeBrowsing, setSafeBrowsing] = useState(true);
  const [warnLinks, setWarnLinks] = useState(true);

  return (
    <div className="min-h-screen bg-gray-100 py-10 px-4">
      <div className="max-w-3xl mx-auto">

        <h1 className="text-4xl font-bold text-center mb-8">
          ⚙️ Settings
        </h1>

        {/* Appearance */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-5">Appearance</h2>

          <SettingToggle
            icon={<FaMoon />}
            title="Dark Mode"
            checked={darkMode}
            onChange={() => setDarkMode(!darkMode)}
          />
        </div>

        {/* Notifications */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-5">Notifications</h2>

          <SettingToggle
            icon={<FaBell />}
            title="Enable Notifications"
            checked={notifications}
            onChange={() => setNotifications(!notifications)}
          />
        </div>

        {/* Scanner */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-5">Scanner</h2>

          <SettingToggle
            icon={<FaCamera />}
            title="Auto Scan"
            checked={autoScan}
            onChange={() => setAutoScan(!autoScan)}
          />

          <div className="flex justify-between items-center mt-5">
            <span className="font-medium">
              Camera Resolution
            </span>

            <select className="border rounded-lg px-3 py-2">
              <option>Low</option>
              <option>Medium</option>
              <option>High</option>
            </select>
          </div>
        </div>

        {/* Security */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-5">
            Security
          </h2>

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
        </div>

        {/* Data */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-5">Data</h2>

          <button className="w-full bg-red-500 hover:bg-red-600 text-white rounded-xl py-3 mb-3 flex items-center justify-center gap-2">
            <FaTrash />
            Clear Scan History
          </button>

          <button className="w-full bg-blue-500 hover:bg-blue-600 text-white rounded-xl py-3 flex items-center justify-center gap-2">
            <FaFileExport />
            Export Scan History
          </button>
        </div>

        {/* About */}
        <div className="bg-white rounded-2xl shadow-lg p-6">
          <h2 className="text-xl font-bold mb-5">About</h2>

          <div className="space-y-4">

            <AboutRow
              icon={<FaInfoCircle />}
              label="Version"
              value="1.0.0"
            />

            <AboutRow
              icon={<FaUserFriends />}
              label="Development Team"
              value="QR Shield Team"
            />

            <AboutRow
              icon={<FaInfoCircle />}
              label="Privacy Policy"
              value="View"
            />

            <AboutRow
              icon={<FaInfoCircle />}
              label="Terms & Conditions"
              value="View"
            />

          </div>
        </div>

      </div>
    </div>
  );
}

type ToggleProps = {
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
}: ToggleProps) {
  return (
    <div className="flex justify-between items-center py-3 border-b last:border-none">
      <div className="flex items-center gap-3 text-lg">
        {icon}
        <span>{title}</span>
      </div>

      <label className="relative inline-flex items-center cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={onChange}
          className="sr-only peer"
        />

        <div className="w-11 h-6 bg-gray-300 rounded-full peer peer-checked:bg-blue-600 after:content-[''] after:absolute after:left-[2px] after:top-[2px] after:bg-white after:h-5 after:w-5 after:rounded-full after:transition-all peer-checked:after:translate-x-5"></div>
      </label>
    </div>
  );
}

type AboutRowProps = {
  icon: React.ReactNode;
  label: string;
  value: string;
};

function AboutRow({
  icon,
  label,
  value,
}: AboutRowProps) {
  return (
    <div className="flex justify-between items-center">
      <div className="flex items-center gap-3">
        {icon}
        <span>{label}</span>
      </div>

      <span className="text-gray-600">{value}</span>
    </div>
  );
}