import { useRef } from "react";
import { FaUpload } from "react-icons/fa";

interface Props {
  onImageSelect: (file: File) => void;
}

export default function ImageUploader({ onImageSelect }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];

    if (file) {
      onImageSelect(file);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-10 text-center">

      <FaUpload
        className="mx-auto text-blue-600 mb-4"
        size={45}
      />

      <p className="mb-6 text-gray-600">
        Upload a QR code image for phishing analysis
      </p>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleChange}
      />

      <button
        onClick={() => inputRef.current?.click()}
        className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl"
      >
        Choose Image
      </button>

    </div>
  );
}