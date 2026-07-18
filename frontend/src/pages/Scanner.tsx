import { useState } from "react";
import CameraScanner from "../components/scanner/CameraScanner";
import { FaCheckCircle } from "react-icons/fa";

export default function Scanner() {
  const [result, setResult] = useState("");

  return (
    <div className="min-h-screen bg-gray-100 py-10 px-6">

      <div className="max-w-5xl mx-auto">

        <h1 className="text-4xl font-bold text-center mb-3">
          QR Code Scanner
        </h1>

        <p className="text-center text-gray-500 mb-10">
          Scan a QR code to check if it is safe.
        </p>

        <div className="bg-white rounded-3xl shadow-xl p-8">

          <div className="border-4 border-dashed border-blue-500 rounded-3xl p-5">

            <CameraScanner
              onScan={(text) => setResult(text)}
            />

          </div>

          {result && (

            <div className="mt-8 bg-green-50 rounded-2xl p-6">

              <div className="flex items-center gap-3">

                <FaCheckCircle className="text-green-600 text-2xl"/>

                <div>

                  <h2 className="font-bold">
                    QR Detected
                  </h2>

                  <p className="text-gray-600 break-all">
                    {result}
                  </p>

                </div>

              </div>

            </div>

          )}

        </div>

      </div>

    </div>
  );
}