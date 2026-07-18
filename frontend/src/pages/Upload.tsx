import { useState } from "react";
import { FaCloudUploadAlt } from "react-icons/fa";

export default function Upload() {

  const [preview, setPreview] = useState("");

  const handleImage = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {

    const file = e.target.files?.[0];

    if (file) {

      setPreview(URL.createObjectURL(file));

    }

  };

  return (

    <div className="min-h-screen bg-gray-100 py-10">

      <div className="max-w-4xl mx-auto bg-white rounded-3xl shadow-xl p-10">

        <h1 className="text-4xl font-bold mb-8">
          Upload QR Image
        </h1>

        <label
          className="cursor-pointer border-4 border-dashed border-blue-500 rounded-3xl h-72 flex flex-col justify-center items-center hover:bg-blue-50 transition"
        >

          <FaCloudUploadAlt
            className="text-6xl text-blue-500"
          />

          <p className="mt-4 text-gray-600">
            Click to Upload QR Image
          </p>

          <input
            hidden
            type="file"
            accept="image/*"
            onChange={handleImage}
          />

        </label>

        {preview && (

          <img
            src={preview}
            className="mt-8 rounded-xl shadow-lg mx-auto max-h-80"
          />

        )}

      </div>

    </div>

  );

}