import { Link } from "react-router-dom";
import {
  FaCamera,
  FaUpload,
  FaShieldAlt,
  FaHistory,
} from "react-icons/fa";

export default function Home() {
  return (
    <div className="bg-slate-100 min-h-screen">

      <section className="max-w-6xl mx-auto py-16 px-6">

        <div className="text-center">

          <FaShieldAlt
            className="mx-auto text-blue-600 mb-6"
            size={70}
          />

          <h1 className="text-5xl font-bold">
            QR Shield
          </h1>

          <p className="mt-4 text-gray-600 text-lg">
            Detect phishing and malicious QR codes safely.
          </p>

        </div>

        <div className="grid md:grid-cols-3 gap-8 mt-16">

          <Link
            to="/scanner"
            className="bg-white rounded-2xl shadow-lg p-8 hover:scale-105 transition"
          >
            <FaCamera size={40} className="text-blue-600 mb-4" />
            <h2 className="text-2xl font-semibold">Live Scanner</h2>
            <p className="mt-2 text-gray-500">
              Scan QR codes using your device camera.
            </p>
          </Link>

          <Link
            to="/upload"
            className="bg-white rounded-2xl shadow-lg p-8 hover:scale-105 transition"
          >
            <FaUpload size={40} className="text-green-600 mb-4" />
            <h2 className="text-2xl font-semibold">Upload Image</h2>
            <p className="mt-2 text-gray-500">
              Upload a QR code image for analysis.
            </p>
          </Link>

          <Link
            to="/history"
            className="bg-white rounded-2xl shadow-lg p-8 hover:scale-105 transition"
          >
            <FaHistory size={40} className="text-purple-600 mb-4" />
            <h2 className="text-2xl font-semibold">History</h2>
            <p className="mt-2 text-gray-500">
              View your previous scan results.
            </p>
          </Link>

        </div>

      </section>

    </div>
  );
}